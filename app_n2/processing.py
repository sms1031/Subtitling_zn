import subprocess
import imageio_ffmpeg
import re
import os
from pydub import AudioSegment
from tqdm import tqdm
from openai import OpenAI
import requests
import math
import time
import openai
from typing import Tuple, Optional
# from logger import logger
from logger import logger
from config import CHUNK_LENGTH_MS, TRANSLATION_CHUNK_LINES, MAX_RETRIES, RETRY_DELAY

def video_to_mp3_with_progress(video_path: str, output_path: str) -> bool:
    """
    将视频文件转换为 MP3 并显示进度条。
    
    Args:
        video_path: 输入视频文件路径
        output_path: 输出MP3文件路径
        
    Returns:
        bool: True表示成功，False表示没有音频流
        
    Raises:
        Exception: 转换过程中的其他错误
    """
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    logger.info(f"使用FFmpeg路径: {ffmpeg_path}")

    # 检查FFmpeg可执行文件是否存在
    if not os.path.exists(ffmpeg_path):
        error_msg = f"FFmpeg可执行文件不存在: {ffmpeg_path}"
        logger.error(error_msg)
        raise Exception(error_msg)

    try:
        # 检查视频文件是否包含音频流
        result = subprocess.run(
            [ffmpeg_path, '-i', video_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False
        )
        
        stderr_output = result.stderr.decode('utf-8')
        
        # 检查是否存在音频流
        if not any(keyword in stderr_output for keyword in ["Stream #0:0", "Stream #0:1", "Audio:"]):
            logger.warning("视频文件不包含音频流")
            return False
            
    except FileNotFoundError as e:
        error_msg = f"FFmpeg可执行文件未找到: {e}"
        logger.error(error_msg)
        raise Exception(error_msg)
    except Exception as e:
        error_msg = f"检查视频音频流时发生错误: {e}"
        logger.error(error_msg)
        raise Exception(error_msg)

    # 获取视频总时长
    probe_command = [ffmpeg_path, '-i', video_path]
    probe_process = subprocess.run(
        probe_command, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        check=False
    )
    probe_stderr = probe_process.stderr.decode('utf-8')

    duration_match = re.search(r'Duration: (\d+):(\d+):(\d+\.\d+)', probe_stderr)
    if duration_match:
        hours, minutes, seconds = map(float, duration_match.groups())
        total_duration = hours * 3600 + minutes * 60 + seconds
        logger.info(f"视频总时长: {total_duration:.2f}秒")
    else:
        error_msg = "无法获取视频时长，请确认视频文件格式正确且未损坏"
        logger.error(error_msg)
        raise Exception(error_msg)

    # FFmpeg转换命令
    command = [
        ffmpeg_path,
        '-i', video_path,
        '-vn',  # 不处理视频
        '-acodec', 'libmp3lame',  # 使用MP3编码器
        '-y',  # 覆盖输出文件
        output_path
    ]

    # 执行转换并显示进度
    process = subprocess.Popen(
        command, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT
    )

    with tqdm(total=total_duration, unit='秒', desc="转换视频为MP3") as pbar:
        while True:
            output_line = process.stdout.readline()
            if output_line == b'' and process.poll() is not None:
                break
            if output_line:
                decoded_line = output_line.decode('utf-8', errors='ignore')
                time_match = re.search(r'time=(\d+):(\d+):(\d+\.\d+)', decoded_line)
                if time_match:
                    hours, minutes, seconds = map(float, time_match.groups())
                    current_time = hours * 3600 + minutes * 60 + seconds
                    pbar.update(current_time - pbar.n)

    # 检查转换结果
    process.wait()
    if process.returncode != 0:
        error_msg = f"FFmpeg转换失败，错误码: {process.returncode}"
        logger.error(error_msg)
        raise Exception(error_msg)

    logger.info(f"视频已成功转换为MP3: {output_path}")
    return True


def mp3_to_wav(mp3_file_path: str, wav_file_path: str) -> None:
    """
    将 MP3 文件转换为 WAV 格式。
    使用FFmpeg直接转换，避免将大文件加载到内存中。
    
    Args:
        mp3_file_path: 输入MP3文件路径
        wav_file_path: 输出WAV文件路径
        
    Raises:
        FileNotFoundError: MP3文件不存在
        Exception: 转换过程中的错误
    """
    logger.info(f"转换MP3到WAV: {mp3_file_path} -> {wav_file_path}")

    if not os.path.exists(mp3_file_path):
        error_msg = f"MP3文件不存在: {mp3_file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
        
    try:
        # 使用FFmpeg直接转换，避免内存问题
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        
        command = [
            ffmpeg_path,
            '-i', mp3_file_path,
            '-acodec', 'pcm_s16le',  # 16位PCM编码
            '-ar', '16000',          # 16kHz采样率（ASR通常使用）
            '-ac', '1',              # 单声道
            '-y',                    # 覆盖输出文件
            wav_file_path
        ]
        
        # 执行转换
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        
        logger.info("MP3转换为WAV成功")
        
    except subprocess.CalledProcessError as e:
        error_msg = f"FFmpeg转换失败: {e.stderr.decode('utf-8', errors='ignore')}"
        logger.error(error_msg)
        raise Exception(error_msg) from e
    except Exception as e:
        error_msg = f"MP3转换为WAV失败: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


def perform_asr_with_retry(chunk_file_path: str, chunk_index: int, total_chunks: int, asr_base_url: str) -> Optional[str]:
    """
    对单个音频块执行ASR，包含重试机制。
    
    Args:
        chunk_file_path: 音频块文件路径
        chunk_index: 当前块索引
        total_chunks: 总块数
        asr_base_url: ASR服务地址
        
    Returns:
        str: ASR结果的SRT内容，失败时返回None
    """
    url = f"{asr_base_url}/asr"
    headers = {"accept": "application/json"}
    
    for attempt in range(MAX_RETRIES):
        try:
            with open(chunk_file_path, "rb") as f:
                files = {"audio_file": (os.path.basename(chunk_file_path), f, "audio/wav")}
                response = requests.post(url, headers=headers, files=files, timeout=60)

            if response.status_code == 200:
                result = response.json()
                srt_content = result.get("srt", "")
                logger.debug(f"ASR成功处理块 {chunk_index + 1}/{total_chunks}")
                return srt_content
            else:
                error_msg = f"ASR请求失败 (块 {chunk_index + 1}/{total_chunks}): HTTP {response.status_code} - {response.text}"
                logger.warning(error_msg)
                if attempt == MAX_RETRIES - 1:
                    raise Exception(error_msg)
                    
        except requests.exceptions.RequestException as e:
            error_msg = f"ASR请求异常 (块 {chunk_index + 1}/{total_chunks}, 尝试 {attempt + 1}/{MAX_RETRIES}): {e}"
            logger.warning(error_msg)
            if attempt == MAX_RETRIES - 1:
                raise Exception(error_msg) from e
        
        # 重试前等待
        if attempt < MAX_RETRIES - 1:
            logger.info(f"等待 {RETRY_DELAY} 秒后重试...")
            time.sleep(RETRY_DELAY)
    
    return None


def perform_asr_and_generate_srt(wav_file_path: str, srt_file_path: str, asr_base_url: str) -> None:
    """
    对 WAV 文件执行 ASR 并生成 SRT 字幕文件。
    通过将大音频文件分块处理来解决内存问题。
    
    Args:
        wav_file_path: 输入WAV文件路径
        srt_file_path: 输出SRT文件路径
        asr_base_url: ASR服务地址
        
    Raises:
        FileNotFoundError: 音频文件不存在
        Exception: ASR处理失败
    """
    if not os.path.exists(wav_file_path):
        error_msg = f"音频文件不存在: {wav_file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    logger.info(f"开始ASR处理: {wav_file_path}")
    
    try:
        audio = AudioSegment.from_wav(wav_file_path)
    except Exception as e:
        error_msg = f"无法加载WAV文件: {e}"
        logger.error(error_msg)
        raise Exception(error_msg) from e
        
    duration_ms = len(audio)
    total_chunks = (duration_ms + CHUNK_LENGTH_MS - 1) // CHUNK_LENGTH_MS
    
    logger.info(f"音频时长: {duration_ms/1000:.2f}秒，将分为 {total_chunks} 个块处理")

    results = []
    current_srt_index = 0
    failed_chunks = []

    with tqdm(total=total_chunks, desc="执行ASR") as pbar:
        for i in range(total_chunks):
            start_ms = i * CHUNK_LENGTH_MS
            end_ms = min(start_ms + CHUNK_LENGTH_MS, duration_ms)
            chunk = audio[start_ms:end_ms]

            chunk_file_path = f"data/chunk_{i}.wav"
            
            try:
                chunk.export(chunk_file_path, format="wav")
                
                srt_chunk_content = perform_asr_with_retry(chunk_file_path, i, total_chunks, asr_base_url)
                
                if srt_chunk_content:
                    # 调整时间偏移
                    adjusted_srt_chunk, new_srt_index = adjust_srt_timestamps(
                        srt_chunk_content, current_srt_index, start_ms / 1000
                    )
                    results.append(adjusted_srt_chunk)
                    current_srt_index = new_srt_index
                else:
                    failed_chunks.append(i + 1)
                    # logger.warning(f"块 {i + 1} ASR失败，将跳过")
                    
            except Exception as e:
                failed_chunks.append(i + 1)
                logger.error(f"处理块 {i + 1} 时发生错误: {e}")
                
            finally:
                # 清理临时块文件
                if os.path.exists(chunk_file_path):
                    try:
                        os.remove(chunk_file_path)
                    except OSError as e:
                        logger.warning(f"删除临时文件失败: {e}")
                        
            pbar.update(1)

    # 检查是否有太多失败的块
    if len(failed_chunks) > total_chunks * 0.5:  # 如果超过50%的块失败
        error_msg = f"ASR处理失败块数过多: {len(failed_chunks)}/{total_chunks}"
        logger.error(error_msg)
        raise Exception(error_msg)
    elif failed_chunks:
        logger.warning(f"部分块ASR失败: {failed_chunks}")

    # 写入SRT文件
    try:
        with open(srt_file_path, "w", encoding="utf-8") as srt_file:
            for chunk_result in results:
                srt_file.write(chunk_result)
        logger.info(f"ASR结果已保存到: {srt_file_path}")
    except Exception as e:
        error_msg = f"写入SRT文件失败: {e}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


def adjust_srt_timestamps(srt_content: str, current_index_offset: int, time_offset_seconds: float) -> Tuple[str, int]:
    """
    根据分块位置调整 SRT 时间戳和序列号。
    
    Args:
        srt_content: 原始SRT内容
        current_index_offset: 当前索引偏移
        time_offset_seconds: 时间偏移（秒）
        
    Returns:
        Tuple[str, int]: 调整后的SRT内容和新的索引偏移
    """
    adjusted_lines = []
    lines = srt_content.strip().split('\n')
    line_idx = 0
    max_index_in_chunk = 0

    def time_to_ms(time_str: str) -> int:
        """将SRT时间格式转换为毫秒"""
        h, m, s_ms = time_str.split(':')
        s, ms = s_ms.split(',')
        return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)

    def ms_to_time(ms: int) -> str:
        """将毫秒转换为SRT时间格式"""
        hours = ms // 3600000
        ms %= 3600000
        minutes = ms // 60000
        ms %= 60000
        seconds = ms // 1000
        milliseconds = ms % 1000
        return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

    while line_idx < len(lines):
        line = lines[line_idx].strip()
        if line.isdigit():
            # 字幕序号行
            original_index = int(line)
            adjusted_lines.append(str(current_index_offset + original_index))
            max_index_in_chunk = max(max_index_in_chunk, original_index)
            line_idx += 1
            
            # 处理时间戳行
            if line_idx < len(lines):
                time_line = lines[line_idx].strip()
                if "-->" in time_line:
                    start_time_str, end_time_str = time_line.split("-->")
                    start_ms = time_to_ms(start_time_str.strip())
                    end_ms = time_to_ms(end_time_str.strip())

                    adjusted_start_ms = start_ms + int(time_offset_seconds * 1000)
                    adjusted_end_ms = end_ms + int(time_offset_seconds * 1000)

                    adjusted_lines.append(f"{ms_to_time(adjusted_start_ms)} --> {ms_to_time(adjusted_end_ms)}")
                    line_idx += 1
                else:
                    adjusted_lines.append(time_line)
                    line_idx += 1
        elif line:
            # 字幕文本行
            adjusted_lines.append(line)
            line_idx += 1
        else:
            # 空行
            adjusted_lines.append("")
            line_idx += 1
            
    adjusted_lines.append("")  # 确保末尾有换行符

    return "\n".join(adjusted_lines), current_index_offset + max_index_in_chunk


def create_openai_client(base_url: str, api_key: str) -> OpenAI:
    """
    创建 OpenAI 客户端实例
    
    Args:
        base_url: OpenAI API地址
        api_key: API密钥
        
    Returns:
        OpenAI: 客户端实例
    """
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=120,  # 增加总体超时时间
        max_retries=0,  # 禁用客户端自动重试，使用我们自己的重试逻辑
    )


def openai_chat_with_retry(user_input: str, client: OpenAI, model: str) -> str:
    """
    向兼容 OpenAI 的本地服务器发送聊天完成请求进行翻译。
    优化的重试机制和错误处理。
    
    Args:
        user_input: 需要翻译的日文文本
        client: OpenAI客户端实例
        model: 模型名称
        
    Returns:
        str: 翻译后的中文文本
        
    Raises:
        Exception: 翻译失败
    """
    prompt = (
        "把日文翻译成中文。保存原有的格式不变，只把日文翻译成中文。"
        "关注最后是有空行或者\\n的换行符，不要丢弃。"
        "保证输入输出行数一致。"
    )

    # 检查输入长度，如果过长则进一步分割
    if len(user_input) > 2000:
        logger.warning(f"输入内容过长 ({len(user_input)} 字符)，将分割处理")
        return split_and_translate(user_input, client, model)

    for attempt in range(MAX_RETRIES):
        try:
            logger.debug(f"翻译尝试 {attempt + 1}/{MAX_RETRIES}")
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_input},
                ],
                temperature=0.1,
                timeout=60,  # 单次请求超时
            )
            translated_content = completion.choices[0].message.content
            logger.debug("翻译成功")
            return translated_content

        except openai.APIConnectionError as e:
            error_msg = f"连接错误: 无法连接到翻译服务器 (尝试 {attempt + 1}/{MAX_RETRIES})"
            logger.warning(f"{error_msg}: {e}")
            
        except openai.APITimeoutError as e:
            error_msg = f"超时错误 (尝试 {attempt + 1}/{MAX_RETRIES}): {e}"
            logger.warning(error_msg)
            
        except openai.APIError as e:
            error_msg = f"API错误 (尝试 {attempt + 1}/{MAX_RETRIES}): {e}"
            logger.warning(error_msg)
            
            # 检查是否是502错误
            if "502" in str(e):
                # 502错误使用更长的等待时间
                if attempt < MAX_RETRIES - 1:
                    wait_time = min(60, RETRY_DELAY * (2 ** attempt))  # 指数退避
                    logger.info(f"检测到502错误，等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                continue
            
        except Exception as e:
            error_msg = f"翻译异常 (尝试 {attempt + 1}/{MAX_RETRIES}): {e}"
            logger.warning(error_msg)

        # 普通重试等待
        if attempt < MAX_RETRIES - 1:
            wait_time = RETRY_DELAY + attempt * 2  # 递增等待时间
            logger.info(f"等待 {wait_time} 秒后重试翻译...")
            time.sleep(wait_time)

    # 所有重试均失败
    error_msg = f"翻译失败: 已达到最大重试次数 ({MAX_RETRIES})"
    logger.error(error_msg)
    raise Exception(error_msg)


def split_and_translate(content: str, client: OpenAI, model: str) -> str:
    """
    将过长的内容分割成小块进行翻译
    
    Args:
        content: 需要翻译的内容
        client: OpenAI客户端实例
        model: 模型名称
        
    Returns:
        str: 翻译后的内容
    """
    lines = content.split('\n')
    result_lines = []
    
    # 按行分组，每组不超过50行或1000字符
    current_group = []
    current_length = 0
    
    for line in lines:
        if len(current_group) >= 50 or current_length + len(line) > 1000:
            # 翻译当前组
            if current_group:
                group_content = '\n'.join(current_group)
                try:
                    translated = translate_small_chunk(group_content, client, model)
                    result_lines.extend(translated.split('\n'))
                except Exception as e:
                    logger.error(f"小块翻译失败，使用原文: {e}")
                    result_lines.extend(current_group)
            
            # 开始新组
            current_group = [line]
            current_length = len(line)
        else:
            current_group.append(line)
            current_length += len(line)
    
    # 处理最后一组
    if current_group:
        group_content = '\n'.join(current_group)
        try:
            translated = translate_small_chunk(group_content, client, model)
            result_lines.extend(translated.split('\n'))
        except Exception as e:
            logger.error(f"最后小块翻译失败，使用原文: {e}")
            result_lines.extend(current_group)
    
    return '\n'.join(result_lines)


def translate_small_chunk(content: str, client: OpenAI, model: str) -> str:
    """
    翻译小块内容，简化的重试机制
    
    Args:
        content: 需要翻译的内容
        client: OpenAI客户端实例
        model: 模型名称
        
    Returns:
        str: 翻译后的内容
    """
    prompt = "把日文翻译成中文。保存原有的格式不变，只把日文翻译成中文。"
    
    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": content},
                ],
                timeout=30,
            )
            return completion.choices[0].message.content
        except Exception as e:
            if attempt < 2:
                time.sleep(3)
                continue
            else:
                # 如果还是失败，返回原文（至少保证程序不崩溃）
                logger.error(f"小块翻译失败，返回原文: {str(e)}")
                return content


def translate_srt_to_chinese(
    input_srt_path: str, 
    output_srt_path: str, 
    openai_base_url: str, 
    openai_api_key: str, 
    openai_model: str
) -> bool:
    """
    使用 OpenAI 兼容服务将 SRT 文件从日文翻译成中文。
    优化版本：更小的批次、更好的错误处理、支持前端配置。
    
    Args:
        input_srt_path: 输入日文SRT文件路径
        output_srt_path: 输出中文SRT文件路径
        openai_base_url: OpenAI API地址
        openai_api_key: OpenAI API密钥
        openai_model: OpenAI模型名称
        
    Returns:
        bool: 翻译是否成功
        
    Raises:
        FileNotFoundError: 输入文件不存在
        Exception: 翻译过程中的错误
    """
    if not os.path.exists(input_srt_path):
        error_msg = f"输入SRT文件不存在: {input_srt_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    logger.info(f"开始翻译SRT文件: {input_srt_path} -> {output_srt_path}")
    logger.info(f"使用配置: URL={openai_base_url}, Model={openai_model}")
    
    # 创建OpenAI客户端
    try:
        client = create_openai_client(openai_base_url, openai_api_key)
    except Exception as e:
        error_msg = f"创建OpenAI客户端失败: {e}"
        logger.error(error_msg)
        raise Exception(error_msg) from e
    
    try:
        with open(input_srt_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
    except Exception as e:
        error_msg = f"读取SRT文件失败: {e}"
        logger.error(error_msg)
        raise Exception(error_msg) from e
    
    total_lines = len(lines)
    
    # 使用更小的批次大小，减少单次请求的负担
    chunk_size = min(80, TRANSLATION_CHUNK_LINES)  # 最多80行一批
    total_steps = math.ceil(total_lines / chunk_size)
    
    logger.info(f"SRT文件共 {total_lines} 行，将分 {total_steps} 块进行翻译（每块最多 {chunk_size} 行）")

    try:
        # 清空输出文件
        if os.path.exists(output_srt_path):
            os.remove(output_srt_path)
            
        for step in tqdm(range(total_steps), desc="翻译字幕", total=total_steps):
            start_index = step * chunk_size
            end_index = min((step + 1) * chunk_size, total_lines)
            
            # 获取当前块的所有行
            chunk_lines = lines[start_index:end_index]
            chunk_content = ''.join(chunk_lines)
            
            if not chunk_content.strip():
                # 如果块内容为空，跳过
                continue
                
            logger.debug(f"正在翻译第 {step + 1}/{total_steps} 块 ({len(chunk_lines)} 行)")
            
            try:
                # 翻译当前块
                translated_content = openai_chat_with_retry(chunk_content, client, openai_model)
                
                # 确保输出格式正确
                if translated_content and not translated_content.endswith('\n'):
                    translated_content += '\n'

                # 追加写入文件
                with open(output_srt_path, "a", encoding="utf-8") as save_file:
                    save_file.write(translated_content)
                
                logger.debug(f"完成翻译块 {step + 1}/{total_steps}")
                
                # 批次间休息，减轻服务器压力
                if step < total_steps - 1:  # 不是最后一块
                    time.sleep(0.5)
                    
            except Exception as e:
                logger.error(f"第 {step + 1} 块翻译失败: {e}")
                
                # 将原文写入，保持文件完整性
                with open(output_srt_path, "a", encoding="utf-8") as save_file:
                    save_file.write(f"# 翻译失败的块 {step + 1}\n")
                    save_file.write(chunk_content)
                    if not chunk_content.endswith('\n'):
                        save_file.write('\n')
                
                # 根据配置决定是否继续或抛出异常
                # 这里选择继续处理，避免因单个块失败导致整个任务失败
                continue
                
    except Exception as e:
        error_msg = f"翻译过程中发生错误: {e}"
        logger.error(error_msg)
        raise Exception(error_msg) from e
    
    logger.info(f"翻译完成，结果已保存到: {output_srt_path}")
    return True