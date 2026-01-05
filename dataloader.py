from modelfunc import *


class SkinLesionDataset(Dataset):
    def __init__(self, img_dir, semantic_dir, instance_dir, size=(256, 256)):
        self.img_dir = img_dir
        self.semantic_dir = semantic_dir
        self.instance_dir = instance_dir

        self.families = sorted([
            os.path.basename(f).split("_semantic.png")[0]
            for f in glob.glob(os.path.join(semantic_dir, "*_semantic.png"))
        ])

        print("Found", len(self.families), "samples")

        self.img_transform = transforms.Compose([
            transforms.Resize(size, interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)      # RGB → [-1,1]
        ])

        self.label_transform = transforms.Compose([
            transforms.Resize(size, interpolation=Image.NEAREST)
        ])

    def load_label(self, path):
        img = Image.open(path).convert("L")
        img = self.label_transform(img)
        arr = np.array(img, dtype=np.int64)         
        return torch.from_numpy(arr)                

    def __getitem__(self, idx):
        family = self.families[idx]

        img_path = os.path.join(self.img_dir, f"{family}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir, f"{family}.png")

        sem_path = os.path.join(self.semantic_dir, f"{family}_semantic.png")
        inst_path = os.path.join(self.instance_dir, f"{family}_instance.png")

        semantic = self.load_label(sem_path)        
        instance = self.load_label(inst_path)     

        img = Image.open(img_path).convert("RGB")
        img = self.img_transform(img)               

        return semantic, instance, img

    def __len__(self):
        return len(self.families)
    

class BrainTumorDataset(Dataset):
    def __init__(self, root_dir, direction='AtoB', transform=None):
        self.root_dir = root_dir
        self.direction = direction
        self.transform = transform
        self.dir_A = os.path.join(root_dir, 'train_A')
        self.dir_B = os.path.join(root_dir, 'train_B')
        self.image_filenames = sorted(os.listdir(self.dir_A))
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256, 246)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            self.mask_transform = transforms.Compose([
                transforms.Resize((256, 256), Image.NEAREST),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ])
    def __len__(self):
        return len(self.image_filenames)
    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        path_A = os.path.join(self.dir_A, img_name)
        path_B = os.path.join(self.dir_B, img_name) 
        image_A = Image.open(path_A).convert('RGB') 
        if os.path.exists(path_B):
            image_B = Image.open(path_B).convert('L') 
        else:
            image_B = Image.new('L', image_A.size, 0)
        A = self.transform(image_A)
        B = self.mask_transform(image_B)
        if self.direction == 'AtoB':
            return {'input': A, 'target': B}
        else:
            return {'input': B, 'target': A}
def get_brain_loader(dataset_path, batch_size=4, direction='AtoB'):
    dataset = BrainTumorDataset(root_dir=dataset_path, direction=direction)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return loader

class SegDataSet(Dataset):
    def __init__(self, base_dir, image_dir, mask_dir, type="lesion"):
        self.base_dir = base_dir
        img_dir = os.path.join(base_dir, image_dir)
        mask_dir = os.path.join(base_dir, mask_dir)
        self.image_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
        self.mask_paths  = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir)]
        self.image_paths.sort()
        self.mask_paths.sort()
    def __getitem__(self, index):
        img_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        transform_img = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
         ])
        transform_mask = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),  
         ])
        image = Image.open(img_path)
        mask = Image.open(mask_path).convert("L")
        image = transform_img(image)
        mask = transform_mask(mask)
        return image, mask
    def __len__(self):
        print(f"Pairs Found {Path(self.base_dir).stem} == {len(self.image_paths)}")
        return len(self.image_paths)
