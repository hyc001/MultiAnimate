import torch, os, imageio, argparse
from torchvision.transforms import v2
from einops import rearrange
import lightning as pl
from lightning.pytorch.strategies import DeepSpeedStrategy
import pandas as pd
from diffsynth import WanVideoPipeline, ModelManager, load_state_dict, load_state_dict_from_folder
from peft import LoraConfig, inject_adapter_in_model
import torchvision
from PIL import Image
import numpy as np
import random
import pickle
from io import BytesIO
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageFilter
import math
import  torch.nn  as nn
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class SelectChannelConv3d(nn.Conv3d):
    """
    A custom 3D convolution layer that dynamically selects and activates specific 
    weight channels based on the idx_list.

    The default `in_channels=8` represents the maximum number of supported 
    characters (7) plus 1 channel for the background. 
    """
    def __init__(self, in_channels=8, out_channels=16, kernel_size=(3,3,3),
                 stride=(1,1,1), padding=(1,1,1), **kwargs):
        super().__init__(in_channels, out_channels,
                         kernel_size=kernel_size, stride=stride,
                         padding=padding, **kwargs)
        
        self.channel_idx = torch.tensor([0, 1, 2], dtype=torch.long)

    def set_channel_idx(self, idx_list):
        # idx_list: Python list
        self.channel_idx = torch.tensor(idx_list, dtype=torch.long, device=self.weight.device)

    def forward(self, x):
        B, C, T, H, W = x.shape
        assert C == 3

        idx = self.channel_idx.to(x.device)
        assert idx.numel() == C

        w_sub = self.weight[:, idx, ...]
        out = F.conv3d(x, w_sub, bias=self.bias,
                       stride=self.stride, padding=self.padding,
                       dilation=self.dilation, groups=1)
        print("conv with 3 channel")
        return out

def build_subdict_with_inchannel_remap(state_dict_all, target_prefix, module_first_conv):
    """
    Extracts a sub-dictionary corresponding to `target_prefix` from the global `state_dict`, 
    and remaps the `in_channels` of the first layer's weights (`0.weight`) to match 
    the shape of `module_first_conv.weight`.

    Remapping rules:
      - If `new_in == old_in`: Use the original weights directly.
      - If `new_in < old_in`: Truncate the weights directly (keeping the first `new_in` channels).
      - If `new_in > old_in`: Tile the original weights along the `in_channel` dimension 
        until the size is >= `new_in`, and then truncate to exactly `new_in`.
        (e.g., `old_in=4, new_in=8` -> `[0, 1, 2, 3, 0, 1, 2, 3]`)

    Note:
        When extending the model to support more characters, we recommend 
        directly copying the weights of the first layer from the original base model 
        to initialize the newly added channels.
    """

    sub = {}
    # extract dictionary
    for k, v in state_dict_all.items():
        if k.startswith(target_prefix):
            sub[k[len(target_prefix):]] = v

    if len(sub) == 0:
        return sub

    if "0.weight" not in sub:
        return sub

    old_w = sub["0.weight"]  # [out_c, old_in, kT, kH, kW]
    new_shape = module_first_conv.weight.shape  # [out_c, new_in, kT, kH, kW]
    out_c_old, old_in, kT, kH, kW = old_w.shape
    out_c_new, new_in, kT_new, kH_new, kW_new = new_shape

    # same shape
    if old_w.shape == new_shape:
        return sub

    # check
    assert out_c_old == out_c_new, f"out_channels mismatch: {out_c_old} vs {out_c_new}"
    assert (kT, kH, kW) == (kT_new, kH_new, kW_new), "kernel size mismatch"

    # new_in <= old_in
    if new_in <= old_in:
        new_w = old_w[:, :new_in, :, :, :].clone()
    else:
        # new_in > old_in
        repeat_times = math.ceil(new_in / old_in)  # 比如 old=4,new=8 -> 2; old=4,new=10 -> 3
        tiled = old_w.repeat(1, repeat_times, 1, 1, 1)  # [out_c, old_in*repeat_times, kT, kH, kW]
        new_w = tiled[:, :new_in, :, :, :].clone()      # 截断到 new_in

    sub["0.weight"] = new_w
    return sub


def generate_label_map(mask_a, mask_b, num_classes=8, thresh=127.5, seed=None):
    """
    Generates a strictly 3-channel label map (Background / Person A / Person B) 
    from two character masks, and returns randomly selected channel indices 
    within the range of 0 to (num_classes - 1) for label embedding channel selection.

    Args:
        mask_a (torch.Tensor): Mask for character A of shape (B, C, T, H, W).
        mask_b (torch.Tensor): Mask for character B of shape (B, C, T, H, W).
        num_classes (int, optional): The maximum number of available channels 
            to sample from. Defaults to 8.
        thresh (float, optional): Binarization threshold for the masks. Defaults to 127.5.
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        Tuple[torch.Tensor, List[int]]:
            - label_map_3ch: The resulting 3-channel label map of shape (B, 3, T, H, W).
            - selected_idx: A list of 3 randomly selected channel indices (e.g., [2, 5, 6]).
    """
    if seed is not None:
        random.seed(seed)

    m_a = (mask_a > thresh).any(dim=1, keepdim=True)   # (B,1,T,H,W)
    m_b = (mask_b > thresh).any(dim=1, keepdim=True)   # (B,1,T,H,W)
    # B, _, T, H, W = m_a.shape

    # random select
    selected_idx = random.sample(range(num_classes), 3)
    # bg_id, a_id, b_id = selected_idx
    
    label_map = torch.zeros_like(m_a, dtype=torch.long)  # (Batch, 1, 81, H, W)
    label_map[m_a == 1] = 1  
    label_map[m_b == 1] = 2  

    label_map_one_hot = torch.nn.functional.one_hot(label_map.squeeze(1), num_classes=3)  # (Batch, 81, H, W, 3)
    print(f"after one hot:{label_map_one_hot.shape}")

    label_map_one_hot = label_map_one_hot.permute(0, 4, 1, 2, 3)  # (Batch, 3, 81, H, W)

    return label_map_one_hot, selected_idx



class TextVideoDataset_onestage(torch.utils.data.Dataset):
    def __init__(self, base_path, max_num_frames=81, frame_interval=2, num_frames=81, height=480, width=832, is_i2v=False,steps_per_epoch=1):
        
        self.file_path = base_path
        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.is_i2v = is_i2v
        self.steps_per_epoch = steps_per_epoch
        
        self.sample_fps = frame_interval
        self.max_frames = max_num_frames
        self.misc_size = [height, width]
        self.video_list = []

            
        file_list = os.listdir(self.file_path)
        print("!!! all dataset length (self_collected_videos_pose): ", len(file_list))
        # 
        for iii_index in file_list:
                
            self.video_list.append(self.file_path+iii_index)

        self.use_pose = True
        print("!!! dataset length: ", len(self.video_list))

        random.shuffle(self.video_list)
            
        self.frame_process = v2.Compose([
            # v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def resize(self, image):
        width, height = image.size
        # 
        image = torchvision.transforms.functional.resize(
            image,
            (self.height, self.width),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        # return torch.from_numpy(np.array(image))
        return image

    # def __getitem__(self, data_id):
    def __getitem__(self, index):
        index = index % len(self.video_list)
        success=False
        for _try in range(5):
            try:
                if _try >0:
                    
                    index = random.randint(1,len(self.video_list))
                    index = index % len(self.video_list)
                
                clean = True
                path_dir = self.video_list[index]

                # main term
                frames_all = pickle.load(open(path_dir+'/frames.pkl','rb'))
                dwpose_all = pickle.load(open(path_dir+'/pose.pkl','rb'))

                # additional term
                mask_female = pickle.load(open(path_dir+'/mask_female.pkl','rb'))
                mask_male = pickle.load(open(path_dir+'/mask_male.pkl','rb'))

                # random sample fps, set as 1
                stride = random.randint(1, self.sample_fps)  
                
                _total_frame_num = len(frames_all)
                cover_frame_num = (stride * self.max_frames)
                max_frames = self.max_frames
                if _total_frame_num < cover_frame_num + 1:
                    start_frame = 0
                    end_frame = _total_frame_num-1
                    stride = max((_total_frame_num//max_frames),1)
                    end_frame = min(stride*max_frames, _total_frame_num-1)
                else:
                    start_frame = random.randint(0, _total_frame_num-cover_frame_num-5)
                    end_frame = start_frame + cover_frame_num

                # print(f"start_frame:{start_frame},end_frame:{end_frame},stride:{stride}")

                frame_list = []
                dwpose_list = []
                female_mask_list = []
                male_mask_list = []
                
                first_frame = None
                first_frame_dwpose = None
                for i_index in range(start_frame, end_frame, stride):
                    i_key = list(frames_all.keys())[i_index]

                    i_frame = Image.open(BytesIO(frames_all[i_key])).convert("RGB")
                    i_dwpose = Image.open(BytesIO(dwpose_all[i_key])).convert("RGB")
                    i_female_mask = Image.open(BytesIO(mask_female[i_key])).convert("RGB")
                    i_male_mask = Image.open(BytesIO(mask_male[i_key])).convert("RGB")
                    
                    if first_frame is None:
                        first_frame=i_frame
                        first_frame_dwpose = i_dwpose

                        frame_list.append(i_frame)
                        dwpose_list.append(i_dwpose)
                        female_mask_list.append(i_female_mask)
                        male_mask_list.append(i_male_mask)

                    else:
                        frame_list.append(i_frame)
                        dwpose_list.append(i_dwpose)
                        female_mask_list.append(i_female_mask)
                        male_mask_list.append(i_male_mask)

                if (end_frame-start_frame) < max_frames:
                    for _ in range(max_frames-(end_frame-start_frame)):
                        i_key = list(frames_all.keys())[end_frame-1]
                        
                        i_frame = Image.open(BytesIO(frames_all[i_key])).convert("RGB")
                        i_dwpose = Image.open(BytesIO(dwpose_all[i_key])).convert("RGB")
                        i_female_mask = Image.open(BytesIO(mask_female[i_key])).convert("RGB")
                        i_male_mask = Image.open(BytesIO(mask_male[i_key])).convert("RGB")
                        
                        frame_list.append(i_frame)
                        dwpose_list.append(i_dwpose)
                        female_mask_list.append(i_female_mask)
                        male_mask_list.append(i_male_mask)

                have_frames = len(frame_list)>0

                if have_frames:
                    l_hight = first_frame.size[1]
                    l_width = first_frame.size[0]

                    # random crop
                    x1 = random.randint(0, l_width//14)
                    x2 = random.randint(0, l_width//14)
                    y1 = random.randint(0, l_hight//14)
                    y2 = random.randint(0, l_hight//14)
                    
                    
                    first_frame = first_frame.crop((x1, y1,l_width-x2, l_hight-y2))               
                    first_frame_tmp = torch.from_numpy(np.array(self.resize(first_frame)))
                    first_frame_dwpose_tmp = torch.from_numpy(np.array(self.resize(first_frame_dwpose.crop((x1, y1,l_width-x2, l_hight-y2))))) 
       
                    video_data_tmp = torch.stack([self.frame_process(self.resize(ss.crop((x1, y1,l_width-x2, l_hight-y2)))) for ss in frame_list], dim=0) # self.transforms(frames)
                    dwpose_data_tmp = torch.stack([torch.from_numpy(np.array(self.resize(ss.crop((x1, y1,l_width-x2, l_hight-y2))))).permute(2,0,1) for ss in dwpose_list], dim=0)
                    female_mask_list_tmp = torch.stack([torch.from_numpy(np.array(self.resize(ss.crop((x1, y1,l_width-x2, l_hight-y2))))).permute(2,0,1) for ss in female_mask_list], dim=0)
                    male_mask_list_tmp = torch.stack([torch.from_numpy(np.array(self.resize(ss.crop((x1, y1,l_width-x2, l_hight-y2))))).permute(2,0,1) for ss in male_mask_list], dim=0)

                video_data = torch.zeros(self.max_frames, 3, self.misc_size[0], self.misc_size[1])
                female_mask_list = torch.zeros(self.max_frames, 3, self.misc_size[0], self.misc_size[1])
                male_mask_list = torch.zeros(self.max_frames, 3, self.misc_size[0], self.misc_size[1])

                dwpose_data = torch.zeros(self.max_frames, 3, self.misc_size[0], self.misc_size[1])
                
                if have_frames:
                    video_data[:len(frame_list), ...] = video_data_tmp      
                    female_mask_list[:len(frame_list), ...] = female_mask_list_tmp      
                    male_mask_list[:len(frame_list), ...] = male_mask_list_tmp

                    dwpose_data[:len(frame_list), ...] = dwpose_data_tmp

                    print(f"sucess read frame {start_frame}")
                    
                video_data = video_data.permute(1,0,2,3) #CTHW
                female_mask_list = female_mask_list.permute(1,0,2,3)
                male_mask_list = male_mask_list.permute(1,0,2,3)

                dwpose_data = dwpose_data.permute(1,0,2,3)
               
                caption = "two person are dancing"
                break
            except Exception as e:
                # 
                caption = "two person are dancing"
                # 
                video_data = torch.zeros(3, self.max_frames, self.misc_size[0], self.misc_size[1])  
                first_frame_tmp = torch.zeros(self.misc_size[0], self.misc_size[1], 3)
                vit_image = torch.zeros(3,self.misc_size[0], self.misc_size[1])
                
                dwpose_data = torch.zeros(3, self.max_frames, self.misc_size[0], self.misc_size[1])  
                # 
                random_ref_dwpose_data = torch.zeros(3, self.max_frames, self.misc_size[0], self.misc_size[1])  
                print('{} read video frame failed with error: {}'.format(path_dir, e))
                continue

        text = caption 
        path = path_dir 
        
        if self.is_i2v:
            video, first_frame = video_data, first_frame_tmp
            data = {"text": text, "path": path, 
                    "first_frame": first_frame, "dwpose_data": dwpose_data, "first_frame_dwpose": first_frame_dwpose_tmp, "video": video, 
                    "female_mask": female_mask_list, "male_mask": male_mask_list
                    }
        else:
            data = {"text": text, "video": video, "path": path}
        return data
    

    def __len__(self):
        
        return len(self.video_list)
 

class LightningModelForTrain_onestage(pl.LightningModule):
    def __init__(
        self,
        dit_path,
        learning_rate=1e-5,
        lora_rank=4, lora_alpha=4, train_architecture="lora", lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming",
        use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,
        pretrained_lora_path=None,
        model_VAE=None,
        # 
    ):
        super().__init__()
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        if os.path.isfile(dit_path):
            model_manager.load_models([dit_path])
        else:
            dit_path = dit_path.split(",")
            model_manager.load_models([dit_path])
        
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)

        self.pipe_VAE = model_VAE.pipe.eval()
        self.tiler_kwargs = model_VAE.tiler_kwargs

        concat_dim = 4
        self.label_embedding = nn.Sequential(
                    SelectChannelConv3d(
                        in_channels=8,
                        out_channels=concat_dim * 4,
                        kernel_size=(3,3,3),
                        stride=(1,1,1),
                        padding=(1,1,1),
                    ),
                    nn.SiLU(),
                    nn.Conv3d(concat_dim * 4, concat_dim * 4, (3,3,3), stride=(1,1,1), padding=(1,1,1)),
                    nn.SiLU(),
                    nn.Conv3d(concat_dim * 4, concat_dim * 4, (3,3,3), stride=(1,1,1), padding=(1,1,1)),
                    nn.SiLU(),
                    nn.Conv3d(concat_dim * 4, concat_dim * 4, (3,3,3), stride=(1,2,2), padding=(1,1,1)),
                    nn.SiLU(),
                    nn.Conv3d(concat_dim * 4, concat_dim * 4, 3, stride=(2,2,2), padding=1),
                    nn.SiLU(),
                    nn.Conv3d(concat_dim * 4, concat_dim * 4, 3, stride=(2,2,2), padding=1),
                    nn.SiLU(),
                    nn.Conv3d(concat_dim * 4, 5120, (1,2,2), stride=(1,2,2), padding=0))

        concat_dim = 4
        self.dwpose_embedding = nn.Sequential(
                    nn.Conv3d(3, concat_dim * 4, (3,3,3), stride=(1,1,1), padding=(1,1,1)),
                    nn.SiLU(),
                    nn.Conv3d(concat_dim * 4, concat_dim * 4, (3,3,3), stride=(1,1,1), padding=(1,1,1)),
                    nn.SiLU(),
                    nn.Conv3d(concat_dim * 4, concat_dim * 4, (3,3,3), stride=(1,1,1), padding=(1,1,1)),
                    nn.SiLU(),
                    nn.Conv3d(concat_dim * 4, concat_dim * 4, (3,3,3), stride=(1,2,2), padding=(1,1,1)),
                    nn.SiLU(),
                    nn.Conv3d(concat_dim * 4, concat_dim * 4, 3, stride=(2,2,2), padding=1),
                    nn.SiLU(),
                    nn.Conv3d(concat_dim * 4, concat_dim * 4, 3, stride=(2,2,2), padding=1),
                    nn.SiLU(),
                    nn.Conv3d(concat_dim * 4, 5120, (1,2,2), stride=(1,2,2), padding=0))

        randomref_dim = 20
        self.randomref_embedding_pose = nn.Sequential(
                    nn.Conv2d(3, concat_dim * 4, 3, stride=1, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=1, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=1, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=2, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=2, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(concat_dim * 4, randomref_dim, 3, stride=2, padding=1),
                    
                    )
        self.freeze_parameters()

        # self.freeze_parameters()
        if train_architecture == "lora":
            self.add_lora_to_model(
                self.pipe.denoising_model(),
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_target_modules=lora_target_modules,
                init_lora_weights=init_lora_weights,
                pretrained_lora_path=pretrained_lora_path,
            )
        else:
            self.pipe.denoising_model().requires_grad_(True)
        
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        
    def freeze_parameters(self):
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()
        self.pipe_VAE.requires_grad_(False)
        self.pipe_VAE.eval()

        self.label_embedding.train()
        self.randomref_embedding_pose.train()
        self.dwpose_embedding.train()
        
        
    def add_lora_to_model(self, model, lora_rank=4, lora_alpha=4, lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming", pretrained_lora_path=None, state_dict_converter=None):
        # Add LoRA to UNet
        self.lora_alpha = lora_alpha
        if init_lora_weights == "kaiming":
            init_lora_weights = True
            
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = inject_adapter_in_model(lora_config, model)
        for param in model.parameters():
            # Upcast LoRA parameters into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)
                
        # Lora pretrained lora weights
        if pretrained_lora_path is not None:
            # 
            try:
                state_dict = load_state_dict(pretrained_lora_path)
            except:
                state_dict = load_state_dict_from_folder(pretrained_lora_path)
            # 
            state_dict_new = {}
            state_dict_new_module = {}
            for key in state_dict.keys():
                
                if 'pipe.dit.' in key:
                    key_new = key.split("pipe.dit.")[1]
                    state_dict_new[key_new] = state_dict[key]
                if "dwpose_embedding" in key or "randomref_embedding_pose" in key or "label_embedding" in key:
                    state_dict_new_module[key] = state_dict[key]
            state_dict = state_dict_new

            state_dict_new = {}
            for key in state_dict_new_module:
                if "dwpose_embedding" in key:
                    state_dict_new[key.split("dwpose_embedding.")[1]] = state_dict_new_module[key]
            self.dwpose_embedding.load_state_dict(state_dict_new, strict=True)

            state_dict_new = {}
            for key in state_dict_new_module:
                if "randomref_embedding_pose" in key:
                    state_dict_new[key.split("randomref_embedding_pose.")[1]] = state_dict_new_module[key]
            self.randomref_embedding_pose.load_state_dict(state_dict_new,strict=True)

            print("Initializing label_embedding weights with previous weights.")
            prefix_fallback = "label_embedding."
            sub_sd = build_subdict_with_inchannel_remap(
                state_dict_new_module,
                target_prefix=prefix_fallback,
                module_first_conv=self.label_embedding[0]
            )
            self.label_embedding.load_state_dict(sub_sd, strict=True)


            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            all_keys = [i for i, _ in model.named_parameters()]
            num_updated_keys = len(all_keys) - len(missing_keys)
            num_unexpected_keys = len(unexpected_keys)
            print(f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected.")
    

    def training_step(self, batch, batch_idx):
        # batch["dwpose_data"]/255.: [1, 3, 81, 832, 480], batch["random_ref_dwpose_data"]/255.: [1, 832, 480, 3]
        text, video, path = batch["text"][0], batch["video"], batch["path"][0]
        female_mask, male_mask = batch["female_mask"], batch["male_mask"] #[1, 3, 81, 832, 480]
        print(f"start training,video:{video.shape}")
        self.pipe_VAE.device = self.device

        label_map, select_index = generate_label_map(female_mask, male_mask)

        label_map = label_map.to(torch.bfloat16)

        # activate chosen channels
        self.label_embedding[0].set_channel_idx(select_index)

       
        label_embedding_output = self.label_embedding((torch.cat([label_map[:,:,:1].repeat(1,1,3,1,1), label_map], dim=2)).to(self.device))
        dwpose_data_tmp = self.dwpose_embedding((torch.cat([batch["dwpose_data"][:,:,:1].repeat(1,1,3,1,1), batch["dwpose_data"]], dim=2)/255.).to(self.device))
        dwpose_data = dwpose_data_tmp + label_embedding_output

        first_frame_dwpose = self.randomref_embedding_pose((batch["first_frame_dwpose"]/255.).to(torch.bfloat16).to(self.device).permute(0,3,1,2)).unsqueeze(2) # [1, 20, 104, 60]

        with torch.no_grad():
            if video is not None:
                # prompt
                # prompt_emb = self.pipe_VAE.encode_prompt(text)
                # we suggect prompt feature cache when trained on A100
                prompt_emb = torch.load("./pos7_emb.pt", map_location="cpu")
                # video
                video = video.to(dtype=self.pipe_VAE.torch_dtype, device=self.pipe_VAE.device)
                latents = self.pipe_VAE.encode_video(video, **self.tiler_kwargs)[0]
                # image
                if "first_frame" in batch: # [1, 853, 480, 3]
                    first_frame = Image.fromarray(batch["first_frame"][0].cpu().numpy())

                    _, _, num_frames, height, width = video.shape
                    image_emb = self.pipe_VAE.encode_image(first_frame, num_frames, height, width)

                else:
                    image_emb = {}
                
                batch = {"latents": latents.unsqueeze(0), "prompt_emb": prompt_emb, "image_emb": image_emb}
        
        # Data
        p1 = random.random()
        p = random.random()
        if p1 < 0.05:
            
            dwpose_data = torch.zeros_like(dwpose_data)

            first_frame_dwpose = torch.zeros_like(first_frame_dwpose)

        latents = batch["latents"].to(self.device)  # [1, 16, 21, 60, 104]
        prompt_emb = batch["prompt_emb"] # batch["prompt_emb"]["context"]:  [1, 1, 512, 4096]
        
        prompt_emb["context"] = prompt_emb["context"].to(self.device)
        print("prompt:",prompt_emb["context"].shape)

        image_emb = batch["image_emb"]
        if "clip_feature" in image_emb:
            image_emb["clip_feature"] = image_emb["clip_feature"].to(self.device) # [1, 257, 1280]
            if p < 0.1:
                image_emb["clip_feature"] = torch.zeros_like(image_emb["clip_feature"]) # [1, 257, 1280]
        if "y" in image_emb:
            
            if p < 0.1:
                image_emb["y"] = torch.zeros_like(image_emb["y"])
            image_emb["y"] = image_emb["y"].to(self.device) + first_frame_dwpose  # [1, 20, 21, 104, 60]

        
        condition =  dwpose_data
        condition = rearrange(condition, 'b c f h w -> b (f h w) c').contiguous()

        # Loss
        self.pipe.device = self.device
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        extra_input = self.pipe.prepare_extra_input(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)

        # Compute loss
        print("main_loss")
        noise_pred = self.pipe.denoising_model()(
            noisy_latents, timestep=timestep, **prompt_emb, **extra_input, **image_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
            add_condition = condition,
        )
        main_loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())

        loss = main_loss * self.pipe.scheduler.training_weight(timestep)

        # Record log
        self.log("train_loss", loss, prog_bar=True)
        return loss


    def configure_optimizers(self):
        trainable_modules = [
            {'params': filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())},
            {'params': self.dwpose_embedding.parameters()},
            {'params': self.randomref_embedding_pose.parameters()},
            {'params': self.label_embedding.parameters()},
        ]
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        return optimizer
    

    # def on_save_checkpoint(self, checkpoint):
    #     checkpoint.clear()
    #     # trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.denoising_model().named_parameters())) + \
    #     #                         list(filter(lambda named_param: named_param[1].requires_grad, self.dwpose_embedding.named_parameters())) + \
    #     #                         list(filter(lambda named_param: named_param[1].requires_grad, self.randomref_embedding_pose.named_parameters()))
    #     trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters())) 
        
    #     trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
    #     # state_dict = self.pipe.denoising_model().state_dict()
    #     state_dict = self.state_dict()
    #     # state_dict.update()
    #     lora_state_dict = {}
    #     for name, param in state_dict.items():
    #         if name in trainable_param_names:
    #             lora_state_dict[name] = param
    #     checkpoint.update(lora_state_dict)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--task",
        type=str,
        default="data_process",
        required=True,
        choices=["data_process", "train"],
        help="Task. `data_process` or `train`.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=True,
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default="./Wan2.1-I2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth",
        help="Path of text encoder.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default="./Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        help="Path of image encoder.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default="./Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth",
        help="Path of VAE.",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default="./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors",
        help="Path of DiT.",
    )
    parser.add_argument(
        "--tiled",
        # default=False,
        default=True,
        action="store_true",
        help="Whether enable tile encode in VAE. This option can reduce VRAM required.",
    )
    parser.add_argument(
        "--tile_size_height",
        type=int,
        default=34,
        help="Tile size (height) in VAE.",
    )
    parser.add_argument(
        "--tile_size_width",
        type=int,
        default=34,
        help="Tile size (width) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_height",
        type=int,
        default=18,
        help="Tile stride (height) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_width",
        type=int,
        default=16,
        help="Tile stride (width) in VAE.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=500,  #500
        help="Number of steps per epoch.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Image width.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q,k,v,o,ffn.0,ffn.2",
        help="Layers with LoRA modules.",
    )
    parser.add_argument(
        "--init_lora_weights",
        type=str,
        default="kaiming",
        choices=["gaussian", "kaiming"],
        help="The initializing method of LoRA weight.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="auto",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
        help="Training strategy",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=4.0,
        help="The weight of the LoRA update matrices.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing_offload",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing offload.",
    )
    parser.add_argument(
        "--train_architecture",
        type=str,
        default="lora",
        choices=["lora", "full"],
        help="Model structure to train. LoRA training or full training.",
    )
    parser.add_argument(
        "--pretrained_lora_path",
        type=str,
        default=None,
        help="Pretrained LoRA path. Required if the training is resumed.",
    )
    parser.add_argument(
        "--use_swanlab",
        default=False,
        action="store_true",
        help="Whether to use SwanLab logger.",
    )
    parser.add_argument(
        "--swanlab_mode",
        default=None,
        help="SwanLab mode (cloud or local).",
    )
    args = parser.parse_args()
    return args


class LightningModelForDataProcess(pl.LightningModule):
    def __init__(self, text_encoder_path, vae_path, image_encoder_path=None, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        super().__init__()
        # model_path = [text_encoder_path, vae_path]
        model_path = [vae_path]
        if image_encoder_path is not None:
            model_path.append(image_encoder_path)
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models(model_path)
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)

        self.tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        
    def test_step(self, batch, batch_idx):
        text, video, path = batch["text"][0], batch["video"], batch["path"][0]
        
        self.pipe.device = self.device
        if video is not None:
            # prompt
            prompt_emb = self.pipe.encode_prompt(text)
            # video
            video = video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            latents = self.pipe.encode_video(video, **self.tiler_kwargs)[0]
            # image
            if "first_frame" in batch:
                first_frame = Image.fromarray(batch["first_frame"][0].cpu().numpy())
                _, _, num_frames, height, width = video.shape
                image_emb = self.pipe.encode_image(first_frame, num_frames, height, width)
            else:
                image_emb = {}
            data = {"latents": latents, "prompt_emb": prompt_emb, "image_emb": image_emb}
            torch.save(data, path + ".tensors.pth")
    
def train_onestage(args):
    
    dataset = TextVideoDataset_onestage(
        args.dataset_path,
        max_num_frames=args.num_frames,
        frame_interval=1,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        is_i2v=args.image_encoder_path is not None,
        steps_per_epoch=args.steps_per_epoch,
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=1,
        num_workers=args.dataloader_num_workers
    )
    model_VAE = LightningModelForDataProcess(
        text_encoder_path=args.text_encoder_path,
        image_encoder_path=args.image_encoder_path,
        vae_path=args.vae_path,
        tiled=args.tiled,
        tile_size=(args.tile_size_height, args.tile_size_width),
        tile_stride=(args.tile_stride_height, args.tile_stride_width),
    )
    model = LightningModelForTrain_onestage(
        dit_path=args.dit_path,
        learning_rate=args.learning_rate,
        train_architecture=args.train_architecture,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        init_lora_weights=args.init_lora_weights,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        pretrained_lora_path=args.pretrained_lora_path,
        model_VAE = model_VAE,
        
    )
    if args.use_swanlab:
        from swanlab.integration.pytorch_lightning import SwanLabLogger
        swanlab_config = {"UPPERFRAMEWORK": "DiffSynth-Studio"}
        swanlab_config.update(vars(args))
        swanlab_logger = SwanLabLogger(
            project="wan", 
            name="wan",
            config=swanlab_config,
            mode=args.swanlab_mode,
            logdir=os.path.join(args.output_path, "swanlog"),
        )
        logger = [swanlab_logger]
    else:
        logger = None
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",     
        precision="bf16",
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        # callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_top_k=-1 ,every_n_train_steps=550)], # save checkpoints every_n_train_steps 
        # callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_top_k=2, monitor="epoch", mode="max" ,every_n_epochs=2, save_on_train_epoch_end=True, save_last=True)], # save checkpoints every_n_train_steps 
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_top_k=2, monitor="epoch", mode="max" ,every_n_epochs=8, save_on_train_epoch_end=True)], # save checkpoints every_n_train_steps 
        logger=logger,
    )
    # trainer.fit(model, dataloader, ckpt_path='./models_multi_labeled1_3p/lightning_logs/version_1/checkpoints/epoch=4-step=1650.ckpt')
    trainer.fit(model, dataloader)

if __name__ == '__main__':
    args = parse_args()
    if args.task == "train":
        train_onestage(args)

