from __future__ import division
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
import numpy as np
import torch
import webdataset as wds
import pytorch_lightning as pl
from torch.utils.data.distributed import DistributedSampler
import SimpleITK as sitk
from util import sobel_3d_edge_detection


class volume(Dataset):
    """Load volumes or latent triplanes and apply optional geometric augmentation."""

    def __init__(self, gt_paths, num_pts=[30000, 30000, 20000, 10000, 10000], sigmas=[1.0, 0.1, 0.01, 0.001, 0.0],
                 latent_paths=None, cond_paths=None, only_latent=False, random_aug=False, fixed_aug=0):
        """Store volume paths, optional latent/condition paths, and sampling settings."""
        super().__init__()
        self.gt_paths = gt_paths
        self.num_pts = num_pts
        self.sigmas = sigmas
        
        self.latent_paths = latent_paths
        self.cond_paths   = cond_paths
        self.only_latent = only_latent
        self.random_aug = random_aug
        self.fixed_aug = fixed_aug
        
    def __len__(self):
        return len(self.gt_paths) if self.fixed_aug==0 else len(self.gt_paths)*self.fixed_aug
    
    def extract_edge_pts(self, path, arr):
        """Load cached edge points or compute and cache them with a 3-D Sobel filter."""
        if not os.path.exists(path):
            edge = sobel_3d_edge_detection(arr.permute(0,3,2,1).contiguous(), device='cpu')
            edge_pts = torch.argwhere(edge>=1).float()
            edge_pts[:,0] = edge_pts[:,0] * 2/arr.shape[3] + 1/arr.shape[3] - 1
            edge_pts[:,1] = edge_pts[:,1] * 2/arr.shape[2] + 1/arr.shape[2] - 1
            edge_pts[:,2] = edge_pts[:,2] * 2/arr.shape[1] + 1/arr.shape[1] - 1
            np.save(path, edge_pts)
        else:
            edge_pts = torch.from_numpy(np.load(path)).float()
        return edge_pts
    
        
    def sample_pts(self, num_pts, sigmas, edge_pts, arr):
        """Sample query points from uniform space and progressively blurred edge bands."""
        pts = []
        try:
            for num, sigma in zip(num_pts, sigmas):
                if num==0:
                    continue
                if sigma==1.0:
                    coord = torch.rand((num*5, 3), dtype=torch.float32) * 2 - 1
                    val = F.grid_sample(arr, coord[None,None,None], padding_mode='border', mode='bilinear')[0,0,0,0]
                    ind = torch.where(val > -1)[0]
                    idx1 = torch.randint(0, ind.shape[0], (int(num*0.8),))
                    idx2 = torch.randint(0, coord.shape[0], (num - int(num*0.8),))
                    coord = torch.cat([coord[ind[idx1]], coord[idx2]], dim=0)
                else:
                    coord = edge_pts[torch.randint(0, edge_pts.shape[0], (num,)), :]
                    coord = coord + torch.randn_like(coord) * sigma  
                pts.append(coord)
        except:
            coord = torch.rand((sum(self.num_pts), 3), dtype=torch.float32) * 2 - 1
            pts = [coord,]
        
        return torch.cat(pts, dim=0)
    
        
    def prepare_rot_flip_aug(self, case, aug_idx):
        """Choose deterministic or random rotation/flip parameters for one sample."""
        if self.fixed_aug:
            if aug_idx>0:
                case['name'] = case['name'] + f'-{aug_idx}'
            flip_axis = aug_idx if aug_idx<=3 else 0
            all_rot   = [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
            rot_axis  = all_rot[max(aug_idx-3, 0)] 
        elif self.random_aug:
            flip_axis = np.random.choice([0,1,2,3],(1,))[0]
            rot_axis  = np.random.permutation([1,2,3])
        else:
            flip_axis, rot_axis = 0, [1,2,3]
        case['flip_axis'], case['rot_axis'] = flip_axis, torch.tensor(rot_axis)
        return case, flip_axis, rot_axis
    
        
    def rot_flip_aug(self, case, flip_axis, rot_axis, arr_key='inputs', pts_key='pts'):
        """Apply the selected spatial transform consistently to arrays and query points."""
        arr = case.get(arr_key)
        pts = case.get(pts_key)
        
        if flip_axis>0:
            if arr!=None: arr = arr.flip(flip_axis)
            flip_axis -= 1
            if pts!=None:
                pts = pts.flip(-1)
                pts[:, flip_axis] = -pts[:, flip_axis]
                pts = pts.flip(-1)
        
        
        if arr!=None: arr = arr.permute(0, *rot_axis)
        if pts!=None:
            rot_axis = (np.array(rot_axis)-1).tolist()
            pts = (pts.flip(-1))[:, rot_axis].flip(-1)
        
        if arr!=None: case[arr_key] = arr
        if pts!=None: case[pts_key] = pts
        
        return case 
    
        
    def __getitem__(self, idx):
        """Load one case, construct supervision, and apply its selected augmentation."""
        case = {}
        
        case_idx = idx % len(self.gt_paths)
        aug_idx  = idx // len(self.gt_paths)
        
        path, cls = list(self.gt_paths[case_idx].items())[0]
        case['path'], case['label'], case['aug_idx'] = path, torch.tensor([cls,]*3).long(), aug_idx
        
        if self.latent_paths!=None:
            if self.fixed_aug and aug_idx>0:
                latent = np.load(self.latent_paths[case_idx].replace('.npy','')+f'-{aug_idx}.npy')
            else:
                latent = np.load(self.latent_paths[case_idx])
            case['latent'] = latent
            case['plane']  = torch.tensor([0,1,2]).long()
            
            if self.cond_paths!=None:
                case['cond']   = np.load(self.cond_paths[case_idx])
            if self.only_latent:
                return case
                
        if path.endswith('.nii.gz'):
            case['name'] = os.path.basename(path).replace('.nii.gz', '')
            img = sitk.ReadImage(path)
            case['spc'], case['org'], case['dct'] = np.array(img.GetSpacing(), dtype=np.float32), np.array(img.GetOrigin(), dtype=np.float32), np.array(img.GetDirection(), dtype=np.float32)
            arr = torch.from_numpy(sitk.GetArrayFromImage(img).transpose(2,1,0).astype(np.float32))[None].clip(0,1)
        elif path.endswith('.npy'):
            case['name'] = os.path.basename(path).replace('.npy', '')
            img = np.load(path, allow_pickle=True).item()
            case['spc'], case['org'], case['dct'] = np.array(img['spc'], dtype=np.float32), np.array(img['org'], dtype=np.float32), np.array(img['dct'], dtype=np.float32)
            arr = torch.from_numpy(img['arr'].transpose(2,1,0).astype(np.float32))[None].clip(0,1)
        else:
            raise NotImplementedError(f"Can not Load data with path: {path}!!!")
        
        arr = 2 * arr - 1
        case['inputs'] = arr
        
        if sum(self.num_pts)>0:
            edge_pts_path = os.path.join(os.path.dirname(path), '{}_sobel_edge_pts.npy'.format(case['name']))
            edge_pts = self.extract_edge_pts(edge_pts_path, case['inputs'])
            
            pts         = self.sample_pts(self.num_pts, self.sigmas, edge_pts, case['inputs'][None])
            case['gt']  = F.grid_sample(case['inputs'][None], pts[None,None,None], padding_mode='border', mode='bilinear')[0,0,0,0]
            case['pts'] = pts
        
        case, flip_axis, rot_axis = self.prepare_rot_flip_aug(case, aug_idx)
        case = self.rot_flip_aug(case, flip_axis, rot_axis, arr_key='inputs', pts_key='pts')
        return case
        

def check_path(path, cls_names_id):
    """Map a dataset path to its configured class id, returning zero if unmatched."""
    for cls_name, cls_id in cls_names_id.items():
        if cls_name in path:
            return cls_id
    return 0
    
def filter_paths(paths, path_include):
    """Keep only paths whose dataset category is listed in ``path_include``."""
    filtered_paths = []
    if path_include==None:
        return paths
    else:
        for p in paths:
            if p.strip('/').split('/')[-3] in path_include:
                filtered_paths.append(p)
        return filtered_paths


def resolve_split_path(path, split_path):
    """Resolve a relative split entry from the project root next to ``data``."""
    path = os.fspath(path)
    if os.path.isabs(path):
        return path

    split_dir = os.path.dirname(os.path.abspath(split_path))
    project_root = os.path.dirname(split_dir)
    project_path = os.path.abspath(os.path.join(project_root, path))
    return project_path if os.path.exists(project_path) else path
            
class Volume(pl.LightningDataModule):
    """Lightning data module for split files, latent conditions, and volume samples."""

    def __init__(self, split_path, num_pts=[30000, 30000, 20000, 10000, 10000], sigmas=[1.0, 0.1, 0.01, 0.001, 0.0], balanced=False, balance_alpha=0.5, balance_max_repeat=5, 
                 cls_names_id=None,
                 overfitting=False,
                 latent_path=None, cond_path=None, only_latent=False, random_aug=False, fixed_aug=0, gen_latent=False, task_path=None, path_include=None, 
                 batch_size=1, val_batch_size=1, num_workers=4, val_num_workers=4, pin_memory=False, distributed=False, seed=None):
        """Prepare train/validation/test records and optional latent/condition paths.

        Args:
            split_path: ``.npy`` split dictionary containing train/val/test paths.
            latent_path: Root directory containing latent ``.npy`` files.
            cond_path: Root directory containing ControlNet condition files.
            task_path: Optional root replacing source paths for test-time tasks.
            path_include: Dataset categories to retain from the split.
            distributed: Use Lightning-compatible distributed samplers in loaders.
        """
        super().__init__()
        
        split = np.load(split_path, allow_pickle=True).item()
        for split_name in ('train', 'val', 'test'):
            split[split_name] = np.array(
                [resolve_split_path(path, split_path) for path in split[split_name]],
                dtype=object,
            )
        if cls_names_id is None:
            cls_names_id = {
                "CTChestAbdomen": 1,
                "CTHeadNeck": 2,
                "CTLegs": 3,
                "MRBody": 4,
                "MRBrain/T1": 5,
                "MRBrain/T2": 6,
            }
        balanced = False if gen_latent else balanced
        
        if balanced:
            group_names = list(dict.fromkeys(cls_name.split('/')[0] for cls_name in cls_names_id))

            if not 0.0 <= balance_alpha <= 1.0:
                raise ValueError(f"balance_alpha must be in [0, 1], got {balance_alpha}.")
            if balance_max_repeat is not None and balance_max_repeat < 1:
                raise ValueError(f"balance_max_repeat must be >= 1 or None, got {balance_max_repeat}.")

            train_paths = list(split["train"])
            grouped = {
                name: [p for p in train_paths if name in p]
                for name in group_names
            }
            others = [
                p for p in train_paths
                if not any(name in p for name in group_names)
            ]

            non_empty = {name: paths for name, paths in grouped.items() if len(paths) > 0}
            if len(non_empty) == 0:
                raise ValueError("balanced=True but no known modality/bodypart group was found.")

            max_len = max(len(paths) for paths in non_empty.values())
            rng = np.random.default_rng(seed)

            balanced_train = []
            balance_report = []

            for name in group_names:
                paths = grouped[name]
                n = len(paths)
                if n == 0:
                    balance_report.append(f"{name}: 0 -> 0")
                    continue

                # alpha=0 gives fully balanced groups; alpha=1 keeps original distribution.
                target_len = int(np.ceil(max_len * ((n / max_len) ** balance_alpha)))
                repeat = int(np.ceil(target_len / n))

                if balance_max_repeat is not None:
                    repeat = min(repeat, int(balance_max_repeat))

                expanded = list(paths) * repeat
                rng.shuffle(expanded)
                expanded = expanded[:target_len]

                balanced_train.extend(expanded)
                balance_report.append(f"{name}: {n} -> {len(expanded)} x{repeat}")

            balanced_train.extend(others)
            rng.shuffle(balanced_train)

            split["train"] = np.array(balanced_train, dtype=object)

            print("Balanced train split:", "; ".join(balance_report))
            if len(others) > 0:
                print(f"Balanced train split: kept {len(others)} unmatched cases once.")
        
        split['train'], split['val'], split['test'] = filter_paths(split['train'], path_include), \
                                                      filter_paths(split['val'], path_include), \
                                                      filter_paths(split['test'], path_include)
        
        self.train_data = [{path:check_path(path, cls_names_id)} for path in split['train']]
        self.val_data   = [{path:check_path(path, cls_names_id)} for path in split['val']]
        self.test_data  = [{path:check_path(path, cls_names_id)} for path in split['test']]
        
        if overfitting:
            overfit_path = os.path.join(
                os.path.dirname(os.path.abspath(split_path)),
                'CTChestAbdomen',
                'selected_200_cases.npy',
            )
            self.train_data = np.load(overfit_path).tolist()
            self.train_data = [{path:check_path(path, cls_names_id)} for path in self.train_data]
            self.val_data, self.test_data = self.train_data, self.train_data
        
        if latent_path!=None:
            self.train_latent = [os.path.join(latent_path, '/'.join(list(i.items())[0][0].strip('/').split('/')[-3:]).replace('.nii.gz', '.npy')) for i in self.train_data]
            self.val_latent   = [os.path.join(latent_path, '/'.join(list(i.items())[0][0].strip('/').split('/')[-3:]).replace('.nii.gz', '.npy')) for i in self.val_data]
            self.test_latent  = [os.path.join(latent_path, '/'.join(list(i.items())[0][0].strip('/').split('/')[-3:]).replace('.nii.gz', '.npy')) for i in self.test_data]
        else:
            self.train_latent, self.val_latent, self.test_latent = None, None, None
        
        if cond_path!=None:
            self.train_cond = [os.path.join(cond_path, '/'.join(list(i.items())[0][0].strip('/').split('/')[-3:]).replace('.nii.gz', '.npy')) for i in self.train_data]
            self.val_cond   = [os.path.join(cond_path, '/'.join(list(i.items())[0][0].strip('/').split('/')[-3:]).replace('.nii.gz', '.npy')) for i in self.val_data]
            self.test_cond  = [os.path.join(cond_path, '/'.join(list(i.items())[0][0].strip('/').split('/')[-3:]).replace('.nii.gz', '.npy')) for i in self.test_data]
        else:
            self.train_cond, self.val_cond, self.test_cond = None, None, None
        
        if gen_latent:
            self.test_data += (self.train_data + self.val_data)
        
        if task_path!=None:
            self.test_data = [{os.path.join(task_path ,'/'.join(list(kv.items())[0][0].strip('/').split('/')[-3:])):list(kv.items())[0][1]} for kv in self.test_data]
            
        self.num_pts = num_pts
        self.sigmas = sigmas
        self.latent_path = latent_path
        self.cond_path = cond_path
        self.only_latent = only_latent
        self.random_aug = random_aug
        self.fixed_aug = fixed_aug
        self.cls_names_id = cls_names_id
        
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.val_num_workers = val_num_workers
        self.pin_memory = pin_memory
        self.distributed = distributed
        self.seed = seed
        
        print("Number of training cases: ", len(self.train_data))
        print("Number of testing  cases: ", len(self.test_data))
        print("Number of val      cases: ", len(self.val_data))
    
    def setup(self, stage):
        """Instantiate the dataset matching Lightning's current fit or test stage."""
        if stage in ['fit']:
            self.train_dataset = volume(self.train_data, self.num_pts, self.sigmas, 
                                        latent_paths=self.train_latent, cond_paths=self.train_cond, only_latent=self.only_latent,
                                        random_aug=self.random_aug, fixed_aug=self.fixed_aug)
            self.val_dataset = volume(self.val_data, self.num_pts, self.sigmas,
                                      latent_paths=self.val_latent, cond_paths=self.val_cond, only_latent=self.only_latent if self.val_cond==None else False,
                                      random_aug=False, fixed_aug=self.fixed_aug)
        else:
            self.test_dataset = volume(self.test_data, [0], [0],
                                      latent_paths=self.test_latent, cond_paths=self.test_cond, only_latent=self.only_latent if self.test_cond==None else False, 
                                      random_aug=False, fixed_aug=self.fixed_aug)
            
    def train_dataloader(self):
        """Build the training loader, using a distributed sampler when requested."""
        if self.distributed:
            self.sampler = DistributedSampler(self.train_dataset, seed=self.seed)
            return wds.WebLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=self.sampler, pin_memory=self.pin_memory)
        else:
            return wds.WebLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=self.pin_memory)
    
    def val_dataloader(self):
        """Build the validation loader with its independently configurable batch size."""
        if self.distributed:
            self.sampler = DistributedSampler(self.val_dataset, shuffle=False)
            return wds.WebLoader(self.val_dataset, batch_size=self.batch_size if self.val_batch_size == None else self.val_batch_size, 
                                 num_workers=self.val_num_workers, shuffle=False, sampler=self.sampler)
        else:
            return wds.WebLoader(self.val_dataset, batch_size=self.batch_size if self.val_batch_size == None else self.val_batch_size, 
                                 num_workers=self.val_num_workers, shuffle=False)

    def test_dataloader(self):
        """Build the test loader with deterministic ordering."""
        if self.distributed:
            self.sampler = DistributedSampler(self.test_dataset, shuffle=False)
            return DataLoader(self.test_dataset, batch_size=self.batch_size if self.val_batch_size == None else self.val_batch_size, 
                                 num_workers=self.val_num_workers, shuffle=False, sampler=self.sampler) 
        else:
            return DataLoader(self.test_dataset, batch_size=self.batch_size if self.val_batch_size == None else self.val_batch_size, 
                              num_workers=self.val_num_workers, shuffle=False) 
