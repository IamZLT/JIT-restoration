import os
import cv2
import glob
import random
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor, Resize, InterpolationMode

from data.degradation_utils import Degradation
from utils.image_utils import random_augmentation, crop_img


class CDD11(Dataset):
    def __init__(self, args, split: str = "train", subset: str = "all"):
        super(CDD11, self).__init__()

        self.args = args
        self.toTensor = ToTensor()
        self.de_type = self.args.de_type
        self.dataset_split = split
        self.subset = subset
        if split == "train":
            self.patch_size = args.patch_size
        else:
            self.patch_size = 64

        self._init()

    def __getitem__(self, index):
        # Randomly select a degradation type
        if self.dataset_split == "train":
            degradation_type = random.choice(list(self.degraded_dict.keys()))
            degraded_image_path = random.choice(self.degraded_dict[degradation_type])
        else:
            degradation_type = self.subset
            degraded_image_path = self.degraded_dict[degradation_type][index]
        
        # Select a degraded image within that type

        degraded_name = os.path.basename(degraded_image_path)

        # Get the corresponding clean image based on the file name
        image_name = os.path.basename(degraded_image_path)
        assert degraded_name == image_name
        clean_image_path = os.path.join(os.path.dirname(self.clean[0]), image_name)

        # Load the images
        #lr = crop_img(np.array(Image.open(degraded_image_path).convert('RGB')), base=16)
        lr = np.array(Image.open(degraded_image_path).convert('RGB'))
        #hr = crop_img(np.array(Image.open(clean_image_path).convert('RGB')), base=16)
        hr = np.array(Image.open(clean_image_path).convert('RGB'))
        # Apply random augmentation and crop
        if self.dataset_split == "train":
            lr, hr = random_augmentation(*self._crop_patch(lr, hr))

        # Convert to tensors
        lr = self.toTensor(lr)
        hr = self.toTensor(hr)

        return [clean_image_path, degradation_type], lr, hr

    def __len__(self):
        return sum(len(images) for images in self.degraded_dict.values())

    def _init(self):
        data_dir = os.path.join(self.args.data_file_dir, "cdd11")
        self.clean = sorted(glob.glob(os.path.join(data_dir, f"{self.dataset_split}/clear", "*.png")))

        if len(self.clean) == 0:
            raise ValueError(f"No clean images found in {os.path.join(data_dir, f'{self.dataset_split}/clear')}")

        self.degraded_dict = {}
        allowed_degradation_folders = self._filter_degradation_folders(data_dir)
        for folder in allowed_degradation_folders:
            folder_name = os.path.basename(folder.strip('/'))
            degraded_images = sorted(glob.glob(os.path.join(folder, "*.png")))
            
            if len(degraded_images) == 0:
                raise ValueError(f"No images found in {folder_name}")
            
            # scale dataset length
            if self.dataset_split == "train":
                degraded_images *= 2
            
            self.degraded_dict[folder_name] = degraded_images

    def _filter_degradation_folders(self, data_dir):
        """
        This function returns folders based on the degradation_type_mode.
        'single', 'double', 'triple', or 'all' degradation types will be returned.
        """
        degradation_folders = sorted(glob.glob(os.path.join(data_dir, self.dataset_split, "*/")))
        filtered_folders = [] 

        for folder in degradation_folders:
            folder_name = os.path.basename(folder.strip('/'))
            if folder_name == "clear":
                continue

            # Count the number of degradations based on the number of underscores in the folder name
            degradation_count = folder_name.count('_') + 1

            # Check the degradation type mode and filter accordingly
            if self.subset == "single" and degradation_count == 1:
                filtered_folders.append(folder)
            elif self.subset == "double" and degradation_count == 2:
                filtered_folders.append(folder)
            elif self.subset == "triple" and degradation_count == 3:
                filtered_folders.append(folder)
            elif self.subset == "all":
                filtered_folders.append(folder)
            # If self.subset is a specific degradation folder name, match it exactly
            elif self.subset not in ["single", "double", "triple", "all"]:
                if folder_name == self.subset:
                    filtered_folders.append(folder)

        print(f"Degradation type mode: {self.subset}")
        print(f"Loading degradation folders: {[os.path.basename(f.strip('/')) for f in filtered_folders]}")
        return filtered_folders

    def _crop_patch(self, img_1, img_2):
        # Crop a patch from both images (degraded and clean) at the same location
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2
    
    
        
class AIOTrainDataset(Dataset):
    """
    Dataset class for training on degraded images.
    """
    def __init__(self, args):
        super(AIOTrainDataset, self).__init__()
        self.args = args
        self.de_temp = 0
        self.de_type = self.args.de_type
        self.D = Degradation(args)
        self.de_dict = {dataset: idx for idx, dataset in enumerate(self.de_type)}
        self.de_dict_reverse = {idx: dataset for idx, dataset in enumerate(self.de_type)}
        
        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])
        self.toTensor = ToTensor()

        self._init_lr()
        self._merge_tasks()
            
    def __getitem__(self, idx):
        lr_sample = self.lr[idx]
        de_id = lr_sample["de_type"]
        deg_type = self.de_dict_reverse[de_id]
        
        if deg_type == "denoise_15" or deg_type == "denoise_25" or deg_type == "denoise_50":
            
            hr = crop_img(np.array(Image.open(lr_sample["img"]).convert('RGB')), base=16)
            hr = self.crop_transform(hr)
            hr = np.array(hr)

            hr = random_augmentation(hr)[0]
            lr = self.D.single_degrade(hr, de_id)
        else:
            if deg_type == "dehaze":
                lr = crop_img(np.array(Image.open(lr_sample["img"]).convert('RGB')), base=16)
                clean_name = self._get_nonhazy_name(lr_sample["img"])
                hr = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
                
            else:
                hr_sample = self.hr[idx]
                lr = crop_img(np.array(Image.open(lr_sample["img"]).convert('RGB')), base=16)
                hr = crop_img(np.array(Image.open(hr_sample["img"]).convert('RGB')), base=16)
        
            lr, hr = random_augmentation(*self._crop_patch(lr, hr))
            
        lr = self.toTensor(lr)
        hr = self.toTensor(hr)
        
        return [lr_sample["img"], de_id], lr, hr
        
    
    def __len__(self):
        return len(self.lr)
    
    
    def _init_lr(self):
        # synthetic datasets
        if 'synllie' in self.de_type:
            self._init_synllie(id=self.de_dict['synllie'])
        if 'deblur' in self.de_type:
            self._init_deblur(id=self.de_dict['deblur'])
        if 'derain' in self.de_type:
            self._init_derain(id=self.de_dict['derain'])
        if 'dehaze' in self.de_type:
            self._init_dehaze(id=self.de_dict['dehaze'])
        if 'denoise_15' in self.de_type:
            self._init_clean(id=0)
        if 'denoise_25' in self.de_type:
            self._init_clean(id=0)
        if 'denoise_50' in self.de_type:
            self._init_clean(id=0)
            
    def _merge_tasks(self):
        self.lr = []
        self.hr = []
        # synthetic datasets
        if "synllie" in self.de_type:
            self.lr += self.synllie_lr
            self.hr += self.synllie_hr
        if "denoise_15" in self.de_type:
            self.lr += self.s15_ids
            self.hr += self.s15_ids
        if "denoise_25" in self.de_type:
            self.lr += self.s25_ids
            self.hr += self.s25_ids
        if "denoise_50" in self.de_type:
            self.lr += self.s50_ids
            self.hr += self.s50_ids
        if "deblur" in self.de_type:
            self.lr += self.deblur_lr 
            self.hr += self.deblur_hr
        if "derain" in self.de_type:
            self.lr += self.derain_lr 
            self.hr += self.derain_hr
        if "dehaze" in self.de_type:
            self.lr += self.dehaze_lr 
            self.hr += self.dehaze_hr

        print(len(self.lr))
   
            
    def _init_synllie(self, id):
        inputs = self.args.data_file_dir + "/llie/LOLv1/Train/input"
        targets = self.args.data_file_dir + "/llie/LOLv1/Train/target"
        
        self.synllie_lr = [{"img" : x, "de_type":id} for x in sorted(glob.glob(inputs + "/*.png"))]
        self.synllie_hr = [{"img" : x, "de_type":id} for x in sorted(glob.glob(targets + "/*.png"))]
        
        self.synllie_counter = 0
        print("Total SynLLIE training pairs : {}".format(len(self.synllie_lr)))
        self.synllie_lr = self.synllie_lr * 20
        self.synllie_hr = self.synllie_hr * 20
        print("Repeated Dataset length : {}".format(len(self.synllie_hr)))
    
    def _init_deblur(self, id):
        """ Initialize the GoPro training dataset """
        inputs = self.args.data_file_dir + "/deblurring/GoPro/crop/train/input_crops/"
        targets = self.args.data_file_dir + "/deblurring/GoPro/crop/train/target_crops/"
        
        self.deblur_lr = [{"img" : x, "de_type":id} for x in sorted(glob.glob(inputs + "/*.png"))]
        self.deblur_hr = [{"img" : x, "de_type":id} for x in sorted(glob.glob(targets + "/*.png"))]
        
        self.deblur_counter = 0
        print("Total Deblur training pairs : {}".format(len(self.deblur_hr)))
        self.deblur_lr = self.deblur_lr * 5
        self.deblur_hr = self.deblur_hr * 5
        print("Repeated Dataset length : {}".format(len(self.deblur_hr)))
        
    def _init_derain(self, id):
        inputs = self.args.data_file_dir + "/deraining/RainTrainL/rainy"
        targets = self.args.data_file_dir + "/deraining/RainTrainL/gt"
        
        self.derain_lr = [{"img" : x, "de_type":id} for x in sorted(glob.glob(inputs + "/*.png"))]
        self.derain_hr = [{"img" : x, "de_type":id} for x in sorted(glob.glob(targets + "/*.png"))]
        
        self.derain_counter = 0
        print("Total Derain training pairs : {}".format(len(self.derain_lr)))
        self.derain_lr = self.derain_lr * 120
        self.derain_hr = self.derain_hr * 120
        print("Repeated Dataset length : {}".format(len(self.derain_hr)))
        
    def _init_dehaze(self, id):
        inputs = self.args.data_file_dir + "/dehazing/RESIDE/"
        targets = self.args.data_file_dir + "/dehazing/RESIDE/clear"
        
        self.dehaze_lr = []
        for part in ["part1", "part2", "part3", "part4"]:
            self.dehaze_lr += [{"img" : x, "de_type":id} for x in sorted(glob.glob(inputs + part + "/*.jpg"))]
        
        self.dehaze_hr = [{"img" : x, "de_type":id} for x in sorted(glob.glob(targets + "/*.jpg"))]
        
        self.dehaze_counter = 0
        print("Total Dehaze training pairs : {}".format(len(self.dehaze_lr)))
        self.dehaze_lr = self.dehaze_lr
        self.dehaze_hr = self.dehaze_hr
        print("Repeated Dataset length : {}".format(len(self.dehaze_lr)))
        
    def _init_clean(self, id):
        inputs = self.args.data_file_dir + "/denoising"
        
        clean = []
        for dataset in ["WaterlooED", "BSD400"]:
            if dataset == "WaterlooED":
                ext = "bmp"
            else:
                ext = "jpg"
            clean += [x for x in sorted(glob.glob(inputs + f"/{dataset}/*.{ext}"))]
            
        if 'denoise_15' in self.de_type:
            self.s15_ids = [{"img": x, "de_type":self.de_dict['denoise_15']} for x in clean]
            self.s15_ids = self.s15_ids * 3
            random.shuffle(self.s15_ids)
            self.s15_counter = 0
        if 'denoise_25' in self.de_type:
            self.s25_ids = [{"img": x, "de_type":self.de_dict['denoise_25']} for x in clean]
            self.s25_ids = self.s25_ids * 3
            random.shuffle(self.s25_ids)
            self.s25_counter = 0
        if 'denoise_50' in self.de_type:
            self.s50_ids = [{"img": x, "de_type":self.de_dict['denoise_50']} for x in clean]
            self.s50_ids = self.s50_ids * 3
            random.shuffle(self.s50_ids)
            self.s50_counter = 0

        self.num_clean = len(clean)
        print("Total Denoise Ids : {}".format(self.num_clean))

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        
        # 检查图像尺寸是否小于 patch_size，如果是则进行填充
        if H < self.args.patch_size or W < self.args.patch_size:
            import numpy as np
            pad_H = max(0, self.args.patch_size - H)
            pad_W = max(0, self.args.patch_size - W)
            
            # 使用反射填充（reflect padding）保持边缘连续性
            # 计算上下左右的填充量
            pad_top = pad_H // 2
            pad_bottom = pad_H - pad_top
            pad_left = pad_W // 2
            pad_right = pad_W - pad_left
            
            # 对两张图像都进行填充
            img_1 = np.pad(img_1, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 
                          mode='reflect')
            img_2 = np.pad(img_2, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 
                          mode='reflect')
            
            H, W = img_1.shape[0], img_1.shape[1]
            print(f"[WARNING] Image too small, padded to {H}x{W}")
        
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2

    def _get_nonhazy_name(self, hazy_name):
        dir_name = os.path.dirname(os.path.dirname(hazy_name)) + "/clear"
        name = hazy_name.split('/')[-1].split('_')[0]
        suffix = os.path.splitext(hazy_name)[1]
        nonhazy_name = dir_name + "/" + name + suffix
        return nonhazy_name
        
    
class IRBenchmarks(Dataset):
    def __init__(self, args):
        super(IRBenchmarks, self).__init__()
        
        self.args = args
        self.benchmarks = args.benchmarks
        self.de_type = self.args.de_type
        self.de_dict = {dataset: idx for idx, dataset in enumerate(self.de_type)}
        
        self.toTensor = ToTensor()
        
        self.resize = Resize(size=(512, 512), interpolation=InterpolationMode.NEAREST)
        
        self._init_lr()
        
    def __getitem__(self, idx):
        lr_sample = self.lr[idx]
        de_id = lr_sample["de_type"]
        
        if "denoise_15" in self.benchmarks or "denoise_25" in self.benchmarks or "denoise_50" in self.benchmarks or "denoise_100" in self.benchmarks or "denoise_75" in self.benchmarks:
            sigma = int(self.benchmarks[-1].split("_")[-1])
            hr = crop_img(np.array(Image.open(lr_sample["img"]).convert('RGB')), base=16)
            lr, _ = self._add_gaussian_noise(hr, sigma)
        else:
            hr_sample = self.hr[idx]
            lr = crop_img(np.array(Image.open(lr_sample["img"]).convert('RGB')), base=16)
            hr = crop_img(np.array(Image.open(hr_sample["img"]).convert('RGB')), base=16)
            
        lr = self.toTensor(lr)
        hr = self.toTensor(hr)
        return [lr_sample["img"], de_id], lr, hr
    
    def __len__(self):
        return len(self.lr)
    
    def _init_lr(self):
        if 'lolv1' in self.benchmarks:
            self._init_synllie(id=self.de_dict['synllie'])
        if 'gopro' in self.benchmarks:
            self._init_deblurring("GoPro", id=self.de_dict['deblur'])
        if 'derain' in self.benchmarks:
            self._init_derain(id=self.de_dict['derain'])
        if 'dehaze' in self.benchmarks:
            self._init_dehaze(id=self.de_dict['dehaze'])
        if 'denoise_15' in self.benchmarks:
            self._init_denoise(id=0)
        if 'denoise_25' in self.benchmarks:
            self._init_denoise(id=0)
        if 'denoise_50' in self.benchmarks:
            self._init_denoise(id=0)

    def _get_nonhazy_name(self, hazy_name):
        dir_name = os.path.dirname(os.path.dirname(hazy_name)) + "/gt"
        name = hazy_name.split('/')[-1].split('_')[0]
        suffix = os.path.splitext(hazy_name)[1]
        nonhazy_name = dir_name + "/" + name + '.png'
        return nonhazy_name
    
    def _add_gaussian_noise(self, clean_patch, sigma):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch
    
    ####################################################################################################
    ## DEBLURRING DATASET
    def _init_deblurring(self, benchmark, id):
        inputs = self.args.data_file_dir + f"/deblurring/{benchmark}/test/input/"
        targets = self.args.data_file_dir + f"/deblurring/{benchmark}/test/target/"
        
        self.lr = [{"img" : x, "de_type":id} for x in sorted(glob.glob(inputs + "/*.png"))]
        self.hr = [{"img" : x, "de_type":id} for x in sorted(glob.glob(targets + "/*.png"))]
        print("Total Deblur testing pairs : {}".format(len(self.hr)))
        
    ####################################################################################################
    ## LLIE DATASET        
    def _init_synllie(self, id):
        inputs = self.args.data_file_dir + "/llie/LOLv1/Test/input"
        targets = self.args.data_file_dir + "/llie/LOLv1/Test/target"
        
        self.lr = [{"img" : x, "de_type":id} for x in sorted(glob.glob(inputs + "/*.png"))]
        self.hr = [{"img" : x, "de_type":id} for x in sorted(glob.glob(targets + "/*.png"))]
        print("Total LLIE testing pairs : {}".format(len(self.hr)))
            
    ####################################################################################################
    ## DERAINING DATASET
    def _init_derain(self, id):
        inputs = self.args.data_file_dir + "/deraining/Rain100L/rainy"
        targets = self.args.data_file_dir + "/deraining/Rain100L/gt"
        
        self.lr = [{"img" : x, "de_type":id} for x in sorted(glob.glob(inputs + "/*.png"))]
        self.hr = [{"img" : x, "de_type":id} for x in sorted(glob.glob(targets + "/*.png"))]
        
        print("Total Derain testing pairs : {}".format(len(self.hr)))
        
    ####################################################################################################
    ## DEHAZING DATASET
    def _init_dehaze(self, id):
        inputs = self.args.data_file_dir + "/dehazing/SOTS/outdoor/hazy"
        targets = self.args.data_file_dir + "/dehazing/SOTS/outdoor/gt"

        self.lr = [{"img" : x, "de_type":id} for x in sorted(glob.glob(inputs + "/*.jpg"))]
        
        self.hr = []
        for sample in self.lr:
            hazy_name = sample["img"]
            clean_name = self._get_nonhazy_name(hazy_name)
            self.hr.append({"img" : clean_name, "de_type":id})
        print("Total Dehazing testing pairs : {}".format(len(self.hr)))
        
    ####################################################################################################
    ## DENOISING DATASET
    def _init_denoise(self, id):
        inputs = self.args.data_file_dir + "/denoising/CBSD68/original_png"
        
        clean = [x for x in sorted(glob.glob(inputs + "/*.png"))]
        
        self.lr = [{"img" : x, "de_type":id} for x in clean]
        self.hr = [{"img" : x, "de_type":id} for x in clean]
        print("Total Denoise testing pairs : {}".format(len(self.lr)))


class SOTSDehazeDataset(Dataset):
    """
    SOTS outdoor dehazing pairs for JiT restoration prototyping.

    Returns:
        meta: [hazy_path, 0]
        lr:   hazy patch  in [0, 1], CxHxW
        hr:   clean patch in [0, 1], CxHxW
    """

    def __init__(self, args, split="train", val_ratio=0.1, seed=0):
        super().__init__()
        self.args = args
        self.split = split
        self.patch_size = args.patch_size
        self.toTensor = ToTensor()

        root = getattr(args, "sots_root", None)
        if root is None:
            root = os.path.join(args.data_file_dir, "dehazing/SOTS/outdoor")
        hazy_dir = os.path.join(root, "hazy")
        gt_dir = os.path.join(root, "gt")

        pairs = []
        for hazy_path in sorted(glob.glob(os.path.join(hazy_dir, "*.jpg"))):
            name = os.path.basename(hazy_path).split("_")[0]
            gt_path = os.path.join(gt_dir, f"{name}.png")
            if os.path.isfile(gt_path):
                pairs.append((hazy_path, gt_path))

        if len(pairs) == 0:
            raise FileNotFoundError(f"No valid SOTS pairs under {root}")

        rng = random.Random(seed)
        rng.shuffle(pairs)
        n_val = max(1, int(len(pairs) * val_ratio))
        if split == "train":
            self.pairs = pairs[n_val:]
        elif split == "val":
            self.pairs = pairs[:n_val]
        else:
            self.pairs = pairs

        print(f"[SOTSDehazeDataset] split={split}, pairs={len(self.pairs)}, crop={self.patch_size}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        hazy_path, gt_path = self.pairs[idx]
        lr = crop_img(np.array(Image.open(hazy_path).convert("RGB")), base=16)
        hr = crop_img(np.array(Image.open(gt_path).convert("RGB")), base=16)

        if self.split == "train":
            lr, hr = random_augmentation(*self._crop_patch(lr, hr))
        else:
            lr, hr = self._center_crop_or_pad(lr, hr)

        return [hazy_path, 0], self.toTensor(lr), self.toTensor(hr)

    def _pad_to_patch(self, img):
        h, w = img.shape[:2]
        pad_h = max(0, self.patch_size - h)
        pad_w = max(0, self.patch_size - w)
        if pad_h == 0 and pad_w == 0:
            return img
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        return np.pad(
            img,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="reflect",
        )

    def _crop_patch(self, img_1, img_2):
        img_1 = self._pad_to_patch(img_1)
        img_2 = self._pad_to_patch(img_2)
        h, w = img_1.shape[:2]
        ind_h = random.randint(0, h - self.patch_size)
        ind_w = random.randint(0, w - self.patch_size)
        return (
            img_1[ind_h : ind_h + self.patch_size, ind_w : ind_w + self.patch_size],
            img_2[ind_h : ind_h + self.patch_size, ind_w : ind_w + self.patch_size],
        )

    def _center_crop_or_pad(self, img_1, img_2):
        img_1 = self._pad_to_patch(img_1)
        img_2 = self._pad_to_patch(img_2)
        h, w = img_1.shape[:2]
        ind_h = (h - self.patch_size) // 2
        ind_w = (w - self.patch_size) // 2
        return (
            img_1[ind_h : ind_h + self.patch_size, ind_w : ind_w + self.patch_size],
            img_2[ind_h : ind_h + self.patch_size, ind_w : ind_w + self.patch_size],
        )


class RESIDEDehazeDataset(SOTSDehazeDataset):
    """RESIDE OTS training set with one random haze level per clean scene.

    OTS contains about 35 hazy variants for every clean image. Treating all
    variants as independent samples makes an epoch unnecessarily large and
    over-represents each scene. This dataset indexes clean scenes and randomly
    selects one corresponding hazy variant whenever a scene is requested.
    """

    def __init__(self, args):
        Dataset.__init__(self)
        self.args = args
        self.split = "train"
        self.patch_size = args.patch_size
        self.toTensor = ToTensor()
        self.repeat = getattr(args, "reside_repeat", 1)

        root = getattr(args, "reside_root", None)
        if root is None:
            root = os.path.join(args.data_file_dir, "dehazing/RESIDE")

        hazy_by_scene = {}
        for part in ("part1", "part2", "part3", "part4"):
            for hazy_path in sorted(glob.glob(os.path.join(root, part, "*.jpg"))):
                scene = os.path.basename(hazy_path).split("_")[0]
                hazy_by_scene.setdefault(scene, []).append(hazy_path)

        samples = []
        for scene, hazy_paths in sorted(hazy_by_scene.items()):
            gt_path = os.path.join(root, "clear", f"{scene}.jpg")
            if os.path.isfile(gt_path):
                samples.append((hazy_paths, gt_path))

        if not samples:
            raise FileNotFoundError(f"No valid RESIDE OTS pairs under {root}")
        if self.repeat < 1:
            raise ValueError(f"reside_repeat must be >= 1, got {self.repeat}")

        self.samples = samples
        num_variants = sum(len(paths) for paths, _ in samples)
        print(
            f"[RESIDEDehazeDataset] scenes={len(samples)}, "
            f"hazy_variants={num_variants}, repeat={self.repeat}, crop={self.patch_size}"
        )

    def __len__(self):
        return len(self.samples) * self.repeat

    def __getitem__(self, idx):
        hazy_paths, gt_path = self.samples[idx % len(self.samples)]
        hazy_path = random.choice(hazy_paths)

        lr = crop_img(np.array(Image.open(hazy_path).convert("RGB")), base=16)
        hr = crop_img(np.array(Image.open(gt_path).convert("RGB")), base=16)
        lr, hr = random_augmentation(*self._crop_patch(lr, hr))

        return [hazy_path, 0], self.toTensor(lr), self.toTensor(hr)


class RainDerainDataset(SOTSDehazeDataset):
    """Paired RainTrainL/Rain100L dataset for standalone deraining."""

    def __init__(self, args, split="train"):
        Dataset.__init__(self)
        self.args = args
        self.split = split
        self.patch_size = args.patch_size
        self.toTensor = ToTensor()

        if split == "train":
            root = getattr(args, "rain_train_root", None)
            if root is None:
                root = os.path.join(args.data_file_dir, "deraining/RainTrainL")
            self.repeat = getattr(args, "rain_repeat", 10)
        elif split in ("val", "test"):
            root = getattr(args, "rain_test_root", None)
            if root is None:
                root = os.path.join(args.data_file_dir, "deraining/Rain100L")
            self.repeat = 1
        else:
            raise ValueError(f"Unknown split: {split}")

        rainy_dir = os.path.join(root, "rainy")
        gt_dir = os.path.join(root, "gt")
        pairs = []
        for rainy_path in sorted(glob.glob(os.path.join(rainy_dir, "*.png"))):
            rainy_name = os.path.basename(rainy_path)
            gt_name = rainy_name.replace("rain-", "norain-", 1)
            gt_path = os.path.join(gt_dir, gt_name)
            if os.path.isfile(gt_path):
                pairs.append((rainy_path, gt_path))

        if not pairs:
            raise FileNotFoundError(f"No valid rain/clean pairs under {root}")
        if self.repeat < 1:
            raise ValueError(f"rain_repeat must be >= 1, got {self.repeat}")

        self.pairs = pairs
        print(
            f"[RainDerainDataset] split={split}, pairs={len(pairs)}, "
            f"repeat={self.repeat}, crop={self.patch_size}"
        )

    def __len__(self):
        return len(self.pairs) * self.repeat

    def __getitem__(self, idx):
        rainy_path, gt_path = self.pairs[idx % len(self.pairs)]
        lr = crop_img(np.array(Image.open(rainy_path).convert("RGB")), base=16)
        hr = crop_img(np.array(Image.open(gt_path).convert("RGB")), base=16)
        lr, hr = self._resize_pair_to_patch(lr, hr)

        if self.split == "train":
            lr, hr = random_augmentation(*self._crop_patch(lr, hr))
        else:
            lr, hr = self._center_crop_or_pad(lr, hr)

        return [rainy_path, 1], self.toTensor(lr), self.toTensor(hr)

    def _resize_pair_to_patch(self, img_1, img_2):
        """Upscale the pair together instead of filling most of 512px with reflection."""
        h, w = img_1.shape[:2]
        if h >= self.patch_size and w >= self.patch_size:
            return img_1, img_2
        scale = max(self.patch_size / h, self.patch_size / w)
        size = (int(round(w * scale)), int(round(h * scale)))
        return (
            cv2.resize(img_1, size, interpolation=cv2.INTER_CUBIC),
            cv2.resize(img_2, size, interpolation=cv2.INTER_CUBIC),
        )