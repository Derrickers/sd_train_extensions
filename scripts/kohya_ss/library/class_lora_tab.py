import gradio as gr
from extensions.sd_train_extensions.scripts.kohya_ss.library.merge_lora_gui import GradioMergeLoRaTab
from extensions.sd_train_extensions.scripts.kohya_ss.library.svd_merge_lora_gui import gradio_svd_merge_lora_tab
from extensions.sd_train_extensions.scripts.kohya_ss.library.verify_lora_gui import gradio_verify_lora_tab
from extensions.sd_train_extensions.scripts.kohya_ss.library.resize_lora_gui import gradio_resize_lora_tab
from extensions.sd_train_extensions.scripts.kohya_ss.library.extract_lora_gui import gradio_extract_lora_tab
from extensions.sd_train_extensions.scripts.kohya_ss.library.extract_lycoris_locon_gui import gradio_extract_lycoris_locon_tab
from extensions.sd_train_extensions.scripts.kohya_ss.library.extract_lora_from_dylora_gui import gradio_extract_dylora_tab
from extensions.sd_train_extensions.scripts.kohya_ss.library.merge_lycoris_gui import gradio_merge_lycoris_tab

# Deprecated code
from extensions.sd_train_extensions.scripts.kohya_ss.library.dataset_balancing_gui import gradio_dataset_balancing_tab
from extensions.sd_train_extensions.scripts.kohya_ss.library.dreambooth_folder_creation_gui import (
    gradio_dreambooth_folder_creation_tab,
)


class LoRATools:
    def __init__(self, folders='', headless: bool = False):
        self.headless = headless
        self.folders = folders

        gr.Markdown(
            'This section provide LoRA tools to help setup your dataset...'
        )
        gradio_extract_dylora_tab(headless=headless)
        gradio_extract_lora_tab(headless=headless)
        gradio_extract_lycoris_locon_tab(headless=headless)
        gradio_merge_lora_tab = GradioMergeLoRaTab()
        gradio_merge_lycoris_tab(headless=headless)
        gradio_svd_merge_lora_tab(headless=headless)
        gradio_resize_lora_tab(headless=headless)
        gradio_verify_lora_tab(headless=headless)
        if folders:
            with gr.Tab('Dataset Preparation'):
                gradio_dreambooth_folder_creation_tab(
                    train_data_dir_input=folders.train_data_dir,
                    reg_data_dir_input=folders.reg_data_dir,
                    output_dir_input=folders.output_dir,
                    logging_dir_input=folders.logging_dir,
                    headless=headless,
                )
                gradio_dataset_balancing_tab(headless=headless)
