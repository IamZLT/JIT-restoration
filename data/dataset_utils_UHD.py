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
        
        # 每个退化类型的重复倍数（可单独配置）
        self.repeat_synllie = getattr(args, 'repeat_synllie', 1)
        self.repeat_deblur = getattr(args, 'repeat_deblur', 1)
        self.repeat_derain = getattr(args, 'repeat_derain', 1)
        self.repeat_dehaze = getattr(args, 'repeat_dehaze', 1)
        self.repeat_denoise = getattr(args, 'repeat_denoise', 1)
        
        print(f"[AIOTrainDataset] Repeat times:")
        print(f"  - synllie (low-light): {self.repeat_synllie}x")
        print(f"  - deblur: {self.repeat_deblur}x")
        print(f"  - derain: {self.repeat_derain}x")
        print(f"  - dehaze: {self.repeat_dehaze}x")
        print(f"  - denoise: {self.repeat_denoise}x")
        
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
            # # 检查是否有预生成的clean图像路径
            if "clean_img" in lr_sample:
                # 使用预生成的noisy/clean对（UHD数据集）
                lr = crop_img(np.array(Image.open(lr_sample["img"]).convert('RGB')), base=16)
                hr = crop_img(np.array(Image.open(lr_sample["clean_img"]).convert('RGB')), base=16)
                lr, hr = random_augmentation(*self._crop_patch(lr, hr))
            # else:
                # 动态生成噪声（原有逻辑）
            # hr = crop_img(np.array(Image.open(lr_sample["img"]).convert('RGB')), base=16)
            # hr = self.crop_transform(hr)
            # hr = np.array(hr)
            # hr = random_augmentation(hr)[0]
            # # 将全局de_id映射到degradation_utils期望的局部索引
            # # degradation_utils中: 0=denoise_15, 1=denoise_25, 2=denoise_50
            # if deg_type == "denoise_15":
            #     local_de_id = 0
            # elif deg_type == "denoise_25":
            #     local_de_id = 1
            # elif deg_type == "denoise_50":
            #     local_de_id = 2
            # else:
            #     local_de_id = de_id  # fallback
            # lr = self.D.single_degrade(hr, local_de_id)
        else:
            # 所有其他任务（deblur, derain, dehaze）都使用lr/hr配对
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
        # UHD Low-Light Enhancement Dataset
        inputs = self.args.data_file_dir + "/llie/UHD_LL/training_set/input"
        targets = self.args.data_file_dir + "/llie/UHD_LL/training_set/gt"
        
        # 支持多种图像格式
        input_imgs = []
        target_imgs = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.JPG']:
            input_imgs += sorted(glob.glob(os.path.join(inputs, ext)))
            target_imgs += sorted(glob.glob(os.path.join(targets, ext)))
        
        self.synllie_lr = [{"img" : x, "de_type":id} for x in input_imgs]
        self.synllie_hr = [{"img" : x, "de_type":id} for x in target_imgs]
        
        self.synllie_counter = 0
        print("Total UHD_LL training pairs : {}".format(len(self.synllie_lr)))
        self.synllie_lr = self.synllie_lr * self.repeat_synllie
        self.synllie_hr = self.synllie_hr * self.repeat_synllie
        print("Repeated Dataset length (x{}): {}".format(self.repeat_synllie, len(self.synllie_hr)))
    
    def _init_deblur(self, id):
        """ Initialize the UHD Deblur training dataset """
        inputs = self.args.data_file_dir + "/deblurring/UHD_deblur/train/input"
        targets = self.args.data_file_dir + "/deblurring/UHD_deblur/train/gt"
        
        # 支持多种图像格式
        input_imgs = []
        target_imgs = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.JPG']:
            input_imgs += sorted(glob.glob(os.path.join(inputs, ext)))
            target_imgs += sorted(glob.glob(os.path.join(targets, ext)))
        
        self.deblur_lr = [{"img" : x, "de_type":id} for x in input_imgs]
        self.deblur_hr = [{"img" : x, "de_type":id} for x in target_imgs]
        
        self.deblur_counter = 0
        print("Total UHD_deblur training pairs : {}".format(len(self.deblur_hr)))
        self.deblur_lr = self.deblur_lr * self.repeat_deblur
        self.deblur_hr = self.deblur_hr * self.repeat_deblur
        print("Repeated Dataset length (x{}): {}".format(self.repeat_deblur, len(self.deblur_hr)))
        
    def _init_derain(self, id):
        # UHD Deraining Dataset  
        inputs = self.args.data_file_dir + "/deraining/UHD_rain/train/input"
        targets = self.args.data_file_dir + "/deraining/UHD_rain/train/gt"
        
        # 支持多种图像格式
        input_imgs = []
        target_imgs = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.JPG']:
            input_imgs += sorted(glob.glob(os.path.join(inputs, ext)))
            target_imgs += sorted(glob.glob(os.path.join(targets, ext)))
        
        self.derain_lr = [{"img" : x, "de_type":id} for x in input_imgs]
        self.derain_hr = [{"img" : x, "de_type":id} for x in target_imgs]
        
        self.derain_counter = 0
        print("Total UHD_rain training pairs : {}".format(len(self.derain_lr)))
        self.derain_lr = self.derain_lr * self.repeat_derain
        self.derain_hr = self.derain_hr * self.repeat_derain
        print("Repeated Dataset length (x{}): {}".format(self.repeat_derain, len(self.derain_hr)))
        
    def _init_dehaze(self, id):
        # UHD Dehazing Dataset
        inputs = self.args.data_file_dir + "/dehazing/UHD_haze/train/input"
        targets = self.args.data_file_dir + "/dehazing/UHD_haze/train/gt"
        
        # 支持多种图像格式
        input_imgs = []
        target_imgs = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.JPG']:
            input_imgs += sorted(glob.glob(os.path.join(inputs, ext)))
            target_imgs += sorted(glob.glob(os.path.join(targets, ext)))
        
        self.dehaze_lr = [{"img" : x, "de_type":id} for x in input_imgs]
        self.dehaze_hr = [{"img" : x, "de_type":id} for x in target_imgs]
        
        self.dehaze_counter = 0
        print("Total UHD_haze training pairs : {}".format(len(self.dehaze_lr)))
        self.dehaze_lr = self.dehaze_lr * self.repeat_dehaze
        self.dehaze_hr = self.dehaze_hr * self.repeat_dehaze
        print("Repeated Dataset length (x{}): {}".format(self.repeat_dehaze, len(self.dehaze_lr)))
        
    def _init_clean(self, id):
        # UHD Denoising Dataset
        # 结构: train/HR/ (clean) + train/UHDN_sigmaXX/ (noisy)
        uhd_noise_base = self.args.data_file_dir + "/denoising/UHD_noise/train"
        hr_dir = os.path.join(uhd_noise_base, "HR")
        
        # 支持多种图像格式
        img_extensions = ['*.png', '*.jpg', '*.jpeg', '*.JPG']
        
        # 加载HR clean图像
        hr_imgs = []
        for ext in img_extensions:
            hr_imgs += sorted(glob.glob(os.path.join(hr_dir, ext)))
        
        # 创建文件名到路径的映射（用于匹配noisy和clean）
        hr_dict = {os.path.basename(img): img for img in hr_imgs}
        
        # Sigma 15
        if 'denoise_15' in self.de_type:
            s15_noisy_dir = os.path.join(uhd_noise_base, "UHDN_sigma15")
            
            s15_noisy = []
            for ext in img_extensions:
                s15_noisy += sorted(glob.glob(os.path.join(s15_noisy_dir, ext)))
            
            self.s15_ids = []
            for noisy_path in s15_noisy:
                noisy_name = os.path.basename(noisy_path)
                # 尝试找到对应的clean图像
                if noisy_name in hr_dict:
                    self.s15_ids.append({
                        "img": noisy_path,
                        "clean_img": hr_dict[noisy_name],
                        "de_type": self.de_dict['denoise_15']
                    })
            
            print("Total UHD_noise sigma15 pairs : {}".format(len(self.s15_ids)))
            self.s15_ids = self.s15_ids * self.repeat_denoise
            random.shuffle(self.s15_ids)
            self.s15_counter = 0
            print("Repeated Dataset length (x{}): {}".format(self.repeat_denoise, len(self.s15_ids)))
        
        # Sigma 25
        if 'denoise_25' in self.de_type:
            s25_noisy_dir = os.path.join(uhd_noise_base, "UHDN_sigma25")
            
            s25_noisy = []
            for ext in img_extensions:
                s25_noisy += sorted(glob.glob(os.path.join(s25_noisy_dir, ext)))
            
            self.s25_ids = []
            for noisy_path in s25_noisy:
                noisy_name = os.path.basename(noisy_path)
                if noisy_name in hr_dict:
                    self.s25_ids.append({
                        "img": noisy_path,
                        "clean_img": hr_dict[noisy_name],
                        "de_type": self.de_dict['denoise_25']
                    })
            
            print("Total UHD_noise sigma25 pairs : {}".format(len(self.s25_ids)))
            self.s25_ids = self.s25_ids * self.repeat_denoise
            random.shuffle(self.s25_ids)
            self.s25_counter = 0
            print("Repeated Dataset length (x{}): {}".format(self.repeat_denoise, len(self.s25_ids)))
        
        # Sigma 50
        if 'denoise_50' in self.de_type:
            s50_noisy_dir = os.path.join(uhd_noise_base, "UHDN_sigma50")
            
            s50_noisy = []
            for ext in img_extensions:
                s50_noisy += sorted(glob.glob(os.path.join(s50_noisy_dir, ext)))
            
            self.s50_ids = []
            for noisy_path in s50_noisy:
                noisy_name = os.path.basename(noisy_path)
                if noisy_name in hr_dict:
                    self.s50_ids.append({
                        "img": noisy_path,
                        "clean_img": hr_dict[noisy_name],
                        "de_type": self.de_dict['denoise_50']
                    })
            
            print("Total UHD_noise sigma50 pairs : {}".format(len(self.s50_ids)))
            self.s50_ids = self.s50_ids * self.repeat_denoise
            random.shuffle(self.s50_ids)
            self.s50_counter = 0
            print("Repeated Dataset length (x{}): {}".format(self.repeat_denoise, len(self.s50_ids)))

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
        """
        Convert hazy image path to clean image path.
        支持UHD_haze结构: train/input/xxx.jpg -> train/gt/xxx.jpg
        """
        # 将 /input/ 替换为 /gt/
        if '/input/' in hazy_name:
            nonhazy_name = hazy_name.replace('/input/', '/gt/')
        else:
            # 旧的RESIDE结构（向后兼容）
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


class UHDTestBenchmarks(Dataset):
    """
    Test dataset for UHD benchmarks (deblur, derain, dehaze, denoise, synllie).
    """
    def __init__(self, args):
        super(UHDTestBenchmarks, self).__init__()
        
        self.args = args
        self.benchmarks = args.benchmarks
        self.de_type = self.args.de_type
        self.de_dict = {dataset: idx for idx, dataset in enumerate(self.de_type)}
        
        self.toTensor = ToTensor()
        
        self._init_lr()
        
    def __getitem__(self, idx):
        lr_sample = self.lr[idx]
        de_id = lr_sample["de_type"]
        
        if "denoise_15" in self.benchmarks or "denoise_25" in self.benchmarks or "denoise_50" in self.benchmarks:
            # For denoise, use clean_img if available
            if "clean_img" in lr_sample:
                lr = np.array(Image.open(lr_sample["img"]).convert('RGB'))
                hr = np.array(Image.open(lr_sample["clean_img"]).convert('RGB'))
            else:
                hr = np.array(Image.open(lr_sample["img"]).convert('RGB'))
                lr = hr.copy()  # For test, use same image
        else:
            hr_sample = self.hr[idx]
            lr = np.array(Image.open(lr_sample["img"]).convert('RGB'))
            hr = np.array(Image.open(hr_sample["img"]).convert('RGB'))
            
        lr = self.toTensor(lr)
        hr = self.toTensor(hr)
        return [lr_sample["img"], de_id], lr, hr
    
    def __len__(self):
        return len(self.lr)
    
    def _init_lr(self):
        if 'synllie' in self.benchmarks or 'uhd_ll' in self.benchmarks:
            self._init_synllie(id=self.de_dict['synllie'])
        if 'deblur' in self.benchmarks or 'uhd_deblur' in self.benchmarks:
            self._init_deblur(id=self.de_dict['deblur'])
        if 'derain' in self.benchmarks or 'uhd_rain' in self.benchmarks:
            self._init_derain(id=self.de_dict['derain'])
        if 'dehaze' in self.benchmarks or 'uhd_haze' in self.benchmarks:
            self._init_dehaze(id=self.de_dict['dehaze'])
        if 'denoise_15' in self.benchmarks:
            self._init_denoise(id=self.de_dict['denoise_15'], sigma=15)
        if 'denoise_25' in self.benchmarks:
            self._init_denoise(id=self.de_dict['denoise_25'], sigma=25)
        if 'denoise_50' in self.benchmarks:
            self._init_denoise(id=self.de_dict['denoise_50'], sigma=50)
    
    def _init_synllie(self, id):
        """UHD Low-Light Enhancement Test Dataset"""
        inputs = self.args.data_file_dir + "/llie/UHD_LL/testing_set/input"
        targets = self.args.data_file_dir + "/llie/UHD_LL/testing_set/gt"
        
        input_imgs = []
        target_imgs = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.JPG']:
            input_imgs += sorted(glob.glob(os.path.join(inputs, ext)))
            target_imgs += sorted(glob.glob(os.path.join(targets, ext)))
        
        self.lr = [{"img" : x, "de_type":id} for x in input_imgs]
        self.hr = [{"img" : x, "de_type":id} for x in target_imgs]
        print("Total UHD_LL test pairs : {}".format(len(self.lr)))
    
    def _init_deblur(self, id):
        """UHD Deblur Test Dataset"""
        inputs = self.args.data_file_dir + "/deblurring/UHD_deblur/test/input300"
        targets = self.args.data_file_dir + "/deblurring/UHD_deblur/test/gt300"
        
        input_imgs = []
        target_imgs = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.JPG']:
            input_imgs += sorted(glob.glob(os.path.join(inputs, ext)))
            target_imgs += sorted(glob.glob(os.path.join(targets, ext)))
        
        self.lr = [{"img" : x, "de_type":id} for x in input_imgs]
        self.hr = [{"img" : x, "de_type":id} for x in target_imgs]
        print("Total UHD_deblur test pairs : {}".format(len(self.lr)))
    
    def _init_derain(self, id):
        """UHD Derain Test Dataset"""
        inputs = self.args.data_file_dir + "/deraining/UHD_rain/test/input"
        targets = self.args.data_file_dir + "/deraining/UHD_rain/test/target"
        
        input_imgs = []
        target_imgs = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.JPG']:
            input_imgs += sorted(glob.glob(os.path.join(inputs, ext)))
            target_imgs += sorted(glob.glob(os.path.join(targets, ext)))
        
        self.lr = [{"img" : x, "de_type":id} for x in input_imgs]
        self.hr = [{"img" : x, "de_type":id} for x in target_imgs]
        print("Total UHD_rain test pairs : {}".format(len(self.lr)))
    
    def _init_dehaze(self, id):
        """UHD Dehaze Test Dataset"""
        inputs = self.args.data_file_dir + "/dehazing/UHD_haze/test/input"
        targets = self.args.data_file_dir + "/dehazing/UHD_haze/test/gt"
        
        input_imgs = []
        target_imgs = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.JPG']:
            input_imgs += sorted(glob.glob(os.path.join(inputs, ext)))
            target_imgs += sorted(glob.glob(os.path.join(targets, ext)))
        
        self.lr = [{"img" : x, "de_type":id} for x in input_imgs]
        self.hr = [{"img" : x, "de_type":id} for x in target_imgs]
        print("Total UHD_haze test pairs : {}".format(len(self.lr)))
    
    def _init_denoise(self, id, sigma):
        """UHD Denoise Test Dataset"""
        uhd_noise_base = self.args.data_file_dir + "/denoising/UHD_noise/test"
        hr_dir = os.path.join(uhd_noise_base, "HR")
        noisy_dir = os.path.join(uhd_noise_base, f"UHDN_sigma{sigma}")
        
        img_extensions = ['*.png', '*.jpg', '*.jpeg', '*.JPG']
        
        hr_imgs = []
        for ext in img_extensions:
            hr_imgs += sorted(glob.glob(os.path.join(hr_dir, ext)))
        
        hr_dict = {os.path.basename(img): img for img in hr_imgs}
        
        noisy_imgs = []
        for ext in img_extensions:
            noisy_imgs += sorted(glob.glob(os.path.join(noisy_dir, ext)))
        
        self.lr = []
        self.hr = []
        for noisy_path in noisy_imgs:
            noisy_name = os.path.basename(noisy_path)
            if noisy_name in hr_dict:
                self.lr.append({
                    "img": noisy_path,
                    "clean_img": hr_dict[noisy_name],
                    "de_type": id
                })
                self.hr.append({
                    "img": hr_dict[noisy_name],
                    "de_type": id
                })
        
        print("Total UHD_noise sigma{} test pairs : {}".format(sigma, len(self.lr)))