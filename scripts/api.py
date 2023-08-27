import asyncio
import json
import os
import shutil
from typing import Optional
import requests
import time

from io import BytesIO

import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel
import requests

import modules.script_callbacks as script_callbacks
from modules import shared, sd_models
from modules.sd_models import checkpoints_loaded
import torch
import gc

from extensions.sd_train_extensions.scripts.kohya_ss.lora_gui import train_model
from extensions.sd_train_extensions.scripts.kohya_ss.finetune import make_captions

from PIL import Image

class LoraParameters(BaseModel):
    breed: list
    trigger: str
    model_name: str
    imgs: list
    pretrained_model_name_or_path:str
    vae:str
    seed: int
    resolution_x: int
    resolution_y: int
    multiple_lr : float
    multiple_epoch : float
    prefix : str
    sdxl: bool
    batch_size: int
    precision: str
    optimizer: str
    optimizer_args: str

class DeleteModelParam(BaseModel):
    name: str

class MoveModelParam(BaseModel):
    name: str
    target: str


def api(_: gr.Blocks, app: FastAPI):
    @app.get("/ping")
    async def ping():
        return "Pong"

    @app.post("/lora/start_training")
    async def start_training(params: LoraParameters):
        start_time = time.time()
        print("start time", start_time)
        print("clear sd_model")
        if not shared.sd_model:
            del shared.sd_model
            checkpoints_loaded.clear()
        end_time = time.time()
        print("unload sd", end_time- start_time)
        code = await train_lora_model(params)
        torch.cuda.empty_cache()
        gc.collect()
        return {"code":code, "desc":"success"}

    # 删除热备
    @app.post("/lora/model_local_delete")
    async def delete_local_model(params: DeleteModelParam):
        try:
            os.remove(os.path.join("models/Lora", params.name+".safetensors"))
        except BaseException as e:
            print(e)
        return {"code": 200}

    # 删除冷备
    @app.post("/lora/model_freeze_delete")
    async def delete_freeze_model(params: MoveModelParam):
        try:
            os.remove(os.path.join(params.target, params.name+".safetensors"))
        except BaseException:
            print("not exist")
        return {"code": 200}

    # 转到冷备
    @app.post("/lora/model_freeze")
    async def freeze_model(params: MoveModelParam):
        try:
            dst = os.path.join(params.target, params.name + ".safetensors")
            if os.path.exists(dst):
                os.remove(dst)
            shutil.move(os.path.join("models/Lora", params.name+".safetensors"), dst)
        except BaseException as e:
            print(e)
        return {"code": 200}

    # 转到热备
    @app.post("/lora/model_freeze_load")
    async def delete_model(params: MoveModelParam):
        try:
            shutil.copyfile(os.path.join(params.target, params.name+".safetensors"), os.path.join("models/Lora", params.name+".safetensors"))
        except BaseException as e:
            print(e)
        return {"code": 200}

    # train lora
    async def train_lora_model(params: LoraParameters):
        if params.vae == "":
            params.vae = None
        if params.resolution_x == 0:
            params.resolution_x = 1024
        if params.resolution_y == 0:
            params.resolution_y = 1024
        if len(params.breed) == 0:
            params.breed = [ 'unknown' ]
        datasetDir = "lora_dataset"
        if not os.path.exists(datasetDir):
            os.mkdir(datasetDir)
        trainDataDir = os.path.join(datasetDir, params.model_name)
        if os.path.exists(trainDataDir):
            shutil.rmtree(trainDataDir)
        imgDataDir = os.path.join(trainDataDir, f"10_{params.breed[0]}")
        os.makedirs(imgDataDir)
        # 做数据集准备
        # 1. 下载图片
        print("downloading img")
        for img in params.imgs:
            response = requests.get(img)
            raw_image = Image.open(BytesIO(response.content))
            imgName = img.split("/")[-1]
            try:
                raw_image.save(os.path.join(imgDataDir, imgName))
            except Exception as e:
                print(e)
        code = 200
        try:
            parser = make_captions.setup_parser()
            args = parser.parse_args([
                imgDataDir,
                "--caption_weights",
                "models/BLIP/model_large_caption.pth",
                "--debug"
            ])
            def post_process(caption : str):
                caption = f" {caption}"
                for breed in params.breed:
                    caption = caption.replace(f" {breed} ", f" {params.trigger} ")
                if params.prefix != "":
                    caption = f"{params.prefix}, {caption}"
                return caption
            make_captions.main(args, post_process)
            # # 开始训练
            train_model(
                headless={"label":False},
                print_only={"label":True},
                pretrained_model_name_or_path=params.pretrained_model_name_or_path,
                v2=False,
                v_parameterization=False,
                sdxl=params.sdxl,
                logging_dir=os.path.join(trainDataDir, "log"),
                train_data_dir=trainDataDir,
                reg_data_dir="",
                output_dir="models/Lora",
                max_resolution=f"{params.resolution_x},{params.resolution_y}",
                learning_rate=1.0,
                lr_scheduler="constant",
                lr_warmup=0,
                train_batch_size=params.batch_size,
                epoch=params.multiple_epoch * 10,
                save_every_n_epochs=params.multiple_epoch * 10,
                mixed_precision=params.precision,
                save_precision=params.precision,
                seed=params.seed,
                num_cpu_threads_per_process=2,
                cache_latents=True,
                cache_latents_to_disk=True,
                caption_extension=".txt",
                enable_bucket=True,
                gradient_checkpointing=True,
                full_fp16=False,
                no_token_padding=False,
                stop_text_encoder_training_pct=0,
                min_bucket_reso=256,
                max_bucket_reso=2048,
                # use_8bit_adam="",
                xformers="xformers",
                save_model_as="safetensors",
                shuffle_caption=False,
                save_state=False,
                resume="",
                prior_loss_weight=1.0,
                text_encoder_lr=1.0 * params.multiple_lr,
                unet_lr=1.0 * params.multiple_lr,
                network_dim=16,
                lora_network_weights='',
                dim_from_weights=False,
                color_aug=False,
                flip_aug=False,
                clip_skip=1,
                gradient_accumulation_steps=1,
                mem_eff_attn=False,
                output_name=params.model_name,
                model_list='custom',  # Keep this. Yes="", it is unused here but required given the common list used
                max_token_length="75",
                max_train_epochs="",
                max_train_steps="",
                max_data_loader_n_workers="0",
                network_alpha=16,
                training_comment="",
                keep_tokens="0",
                lr_scheduler_num_cycles="1",
                lr_scheduler_power="",
                persistent_data_loader_workers=False,
                bucket_no_upscale=True,
                random_crop=False,
                bucket_reso_steps=32,
                caption_dropout_every_n_epochs=0.0,
                caption_dropout_rate=0,
                optimizer=params.optimizer,
                optimizer_args=params.optimizer_args,
                noise_offset_type="Original",
                noise_offset=0,
                adaptive_noise_scale=0,
                multires_noise_iterations=0,
                multires_noise_discount=0,
                LoRA_type="Standard",
                factor=-1,
                use_cp=False,
                decompose_both=False,
                train_on_input=True,
                conv_dim=32,
                conv_alpha=32,
                sample_every_n_steps=0,
                sample_every_n_epochs=0,
                sample_sampler="euler_a",
                sample_prompts="",
                additional_parameters="",
                vae_batch_size=0,
                min_snr_gamma=5,
                down_lr_weight="",
                mid_lr_weight="",
                up_lr_weight="",
                block_lr_zero_threshold="",
                block_dims="",
                block_alphas="",
                conv_block_dims="",
                conv_block_alphas="",
                weighted_captions=False,
                unit=1,
                save_every_n_steps=0,
                save_last_n_steps=0,
                save_last_n_steps_state=0,
                use_wandb=False,
                wandb_api_key="",
                scale_v_pred_loss_like_noise_pred=False,
                scale_weight_norms=5,
                network_dropout=0,
                rank_dropout=0,
                module_dropout=0,
                sdxl_cache_text_encoder_outputs=False,
                sdxl_no_half_vae=True,
                full_bf16=False,
                min_timestep=0,
                max_timestep=1000,
                lr_scheduler_args="",
                v_pred_like_loss=0
            )
        except:
            code = 201
        try:
            shutil.rmtree(trainDataDir)
        except:
            print("delete data error")
        return code


script_callbacks.on_app_started(api)

print("Train API loaded")
