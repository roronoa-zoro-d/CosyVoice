import sys
import os
import json
import random
import time

# 代码目录 环境变量
cosyvoice_root_dir = '/home/zhangjiayuan/program/audio/tts/CosyVoice'


sys.path.append(f'{cosyvoice_root_dir}')
sys.path.append(f'{cosyvoice_root_dir}/third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import torch



# TTS Model
model_dir='/data/nas/models/tts/CosyVoice'

# snapshot_download('iic/CosyVoice2-0.5B', local_dir=f'{model_dir}/pretrained_models/CosyVoice2-0.5B')
cosyvoice = CosyVoice2(f'{model_dir}/pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)


num_gpu = 4
num_wav_per_spk = 100
out_sample_rate=16000
out_dir = '/data/nas/zhangjiayuan/experiment/paraformer_finitune/datas/tts_datas'
# clone-spk
speech_root_dir = '/data/nas/emilia/Amphion___Emilia/raw/ZH'
clone_data_file = 'select_spk.jsonl'

clone_datas = []
with open(clone_data_file, 'r', encoding='utf-8') as file:
    for line in file:
        clone_datas.append(json.loads(line))
print(f'read {len(clone_datas)} clone_spk')
        
# generate txt
txts = []
txt_file = 'finitune_corcups.txt'
with open(txt_file, 'r', encoding='utf-8') as f:
    for line in f:
        txts.append(line.strip())
print(f'read {len(txts)} txt ' )


# 加载音频文件
def load_resample_audio(file_path, target_sample_rate=16000):
    # 使用torchaudio.load加载音频文件，默认使用合适的后端（如FFmpeg）
    waveform, sample_rate = torchaudio.load(file_path)

    # 创建重采样器
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
    
    # 执行重采样
    resampled_waveform = resampler(waveform)

    return resampled_waveform

def generate_audio(text, spk_data, target_sample_rate=16000):
    
    speech_datas = []
    norm_text = ""
    for i, j in enumerate(cosyvoice.inference_zero_shot_with_spk(text, spk_data, stream=False)):
        speech_datas.append(j['tts_speech'])
        norm_text += j['norm_text']
    speech_data = torch.cat(speech_datas, dim=1)
    resampler = torchaudio.transforms.Resample(orig_freq=cosyvoice.sample_rate, new_freq=target_sample_rate)
    resampled_waveform = resampler(speech_data)
    
    return resampled_waveform, norm_text

def save_data(speech_data, txt, prefix, sample_rate=16000):
    torchaudio.save(f'{prefix}.wav', speech_data, sample_rate)
    with open(f'{prefix}.txt', 'w') as f:
        f.write(f"{txt}\n")


if __name__ == '__main__':
    
    gpu_id = eval(sys.argv[1])
    
    batch_data = [clone_datas[i] for i in range(gpu_id, len(clone_datas), num_gpu)]
    print(f'gpu_id:{gpu_id} , select {len(batch_data)} spk for clone')
    
    writer = open(f'log.process.gpu-{gpu_id}.txt', 'w')
    
    total_wav = len(batch_data) * num_wav_per_spk
    wav_count = 0
    
    infer_time = 0
    total_wav_time = 0
    st_time = time.time()
    for i, data in enumerate(batch_data):
        wav_file = "{}/{}".format(speech_root_dir, data['wav'])
        out_spk_dir = "{}/{}".format(out_dir, data['speaker'])
        os.makedirs(out_spk_dir, exist_ok=True)
        
        
        prompt_speech_16k = load_resample_audio(wav_file)
        prompt_text = data['text']    
        clone_spk = data['id']
        
        
        spk_data = cosyvoice.generate_spk_data(prompt_text, prompt_speech_16k)
        prompt_text = spk_data['prompt_ori_text']
        
        tts_texts = random.choices(txts, k=num_wav_per_spk)
        for j, txt in enumerate(tts_texts):
            if len(txt) < 0.5 * len(prompt_text):
                continue
            st0_time = time.time()
            speech_data, norm_text = generate_audio(txt, spk_data, target_sample_rate=out_sample_rate)
            ed0_time = time.time()
            infer_time += ed0_time - st0_time
            total_wav_time += speech_data.shape[1] / cosyvoice.sample_rate
            save_data(speech_data, norm_text, f'{out_spk_dir}/{clone_spk}_{j:04}', sample_rate=out_sample_rate)
            wav_count += 1 
            total_use_time = time.time() - st_time
        
            log_str = f'generate {i}/{len(batch_data)} {j}/{num_wav_per_spk} {wav_count}/{total_wav},'
            log_str += 'use rtf {:.1f}/{:.1f} min={:.3f}, total use  {:.2f} hour'.format(infer_time/60.0, total_wav_time/60.0, infer_time/total_wav_time, total_use_time/3600.0)
            
        
            # print(f'prompt spk {clone_spk}, text {prompt_text}, wav shape {prompt_speech_16k.shape} wav_file {wav_file}')

            if wav_count > 0 and wav_count % 100 == 0:
                print(log_str)
                writer.write(log_str+'\n')
                
    
    writer.close()
