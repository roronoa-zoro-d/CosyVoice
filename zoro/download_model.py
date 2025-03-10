from modelscope import snapshot_download
cosyvoice_root_dir = '/home/zhangjiayuan/program/audio/tts/CosyVoice'
model_dir='/data/nas/models/tts/CosyVoice'

snapshot_download('iic/CosyVoice2-0.5B', local_dir=f'{model_dir}/pretrained_models/CosyVoice2-0.5B')