from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import os
import shutil
import processing
import gradio as gr
import time
import gc
import asyncio
from typing import Optional, Tuple, AsyncGenerator, Union
from logger import logger
from config import (
    TEMP_DIR, OUTPUT_DIR, SUPPORTED_VIDEO_FORMATS,
    DEFAULT_ASR_BASE_URL, DEFAULT_OPENAI_BASE_URL, 
    DEFAULT_OPENAI_API_KEY, DEFAULT_OPENAI_MODEL
)

app = FastAPI(title="视频字幕生成服务", version="1.0.0")

# 确保必要的目录存在
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def cleanup_file(file_path: str, description: str = "") -> None:
    """
    安全地删除文件，包含错误处理。
    
    Args:
        file_path: 要删除的文件路径
        description: 文件描述，用于日志
    """
    if file_path and os.path.exists(file_path):
        try:
            time.sleep(0.1)  # 短暂延迟确保文件不被占用
            os.remove(file_path)
            logger.info(f"已删除{description}: {file_path}")
        except OSError as e:
            logger.warning(f"删除{description}失败 {file_path}: {e}")


def cleanup_chunk_files(temp_dir: str = TEMP_DIR) -> None:
    """
    清理分块临时文件。
    
    Args:
        temp_dir: 临时目录路径
    """
    if not os.path.isdir(temp_dir):
        return
        
    cleaned_count = 0
    for filename in os.listdir(temp_dir):
        if filename.startswith("chunk_") and filename.endswith(".wav"):
            chunk_path = os.path.join(temp_dir, filename)
            try:
                time.sleep(0.1)
                os.remove(chunk_path)
                cleaned_count += 1
                logger.debug(f"已删除分块文件: {filename}")
            except OSError as e:
                logger.warning(f"删除分块文件失败 {filename}: {e}")
    
    if cleaned_count > 0:
        logger.info(f"清理完成，共删除 {cleaned_count} 个分块文件")


def validate_video_file(filename: Optional[str]) -> bool:
    """
    验证视频文件格式是否受支持。
    
    Args:
        filename: 文件名
        
    Returns:
        bool: 是否为支持的格式
    """
    if not filename:
        return False
    return any(filename.lower().endswith(ext) for ext in SUPPORTED_VIDEO_FORMATS)


async def process_video_for_gradio(
    video_file_obj: gr.File,
    asr_base_url: str,
    openai_base_url: str,
    openai_api_key: str,
    openai_model: str
) -> AsyncGenerator[Tuple[str, Union[gr.File, None]], None]:
    """
    异步处理视频文件，生成并翻译字幕，返回中文SRT文件。
    结合版本：异步处理 + 实时状态反馈 + 手动时间记录 + 前端模型配置
    
    Args:
        video_file_obj: Gradio文件对象
        asr_base_url: ASR服务地址
        openai_base_url: OpenAI API地址
        openai_api_key: OpenAI API密钥
        openai_model: OpenAI模型名称
        
    Yields:
        Tuple[str, Union[gr.File, None]]: (状态信息, 输出文件或None)
    """
    if video_file_obj is None:
        yield "❌ 错误：请上传一个视频文件。", None
        return

    # 验证模型配置
    if not asr_base_url.strip():
        yield "❌ 错误：请填写ASR服务地址。", None
        return
    if not openai_base_url.strip():
        yield "❌ 错误：请填写OpenAI API地址。", None
        return
    if not openai_api_key.strip():
        yield "❌ 错误：请填写OpenAI API密钥。", None
        return
    if not openai_model.strip():
        yield "❌ 错误：请填写OpenAI模型名称。", None
        return

    # 记录开始时间
    process_start_time = time.time()
    from datetime import datetime
    start_datetime = datetime.now()
    
    yield f"⏰ 开始处理时间：{start_datetime.strftime('%H:%M:%S')}\n🔧 使用配置：\n  - ASR服务：{asr_base_url}\n  - AI模型：{openai_model}\n等待处理...", None
    await asyncio.sleep(0.1)

    # 获取文件信息
    gradio_temp_video_path = video_file_obj.name
    original_filename = os.path.basename(gradio_temp_video_path)
    
    # 验证文件格式
    if not validate_video_file(original_filename):
        yield f"❌ 错误：不支持的视频格式。请上传: {', '.join(SUPPORTED_VIDEO_FORMATS)}", None
        return

    yield f"📁 文件验证通过：{original_filename}\n正在准备处理...", None
    await asyncio.sleep(0.1)  # 让UI有时间更新

    # 复制到工作目录
    video_path_in_temp_dir = os.path.join(TEMP_DIR, original_filename)
    
    try:
        yield "📋 正在复制文件到工作目录...", None
        await asyncio.sleep(0.1)
        
        # 异步文件复制
        await asyncio.to_thread(shutil.copyfile, gradio_temp_video_path, video_path_in_temp_dir)
        logger.info(f"Gradio临时文件已复制到: {video_path_in_temp_dir}")
        current_video_path = video_path_in_temp_dir
        
        yield "✅ 文件复制完成！", None
        await asyncio.sleep(0.1)
        
    except Exception as e:
        error_msg = f"❌ 无法将视频文件复制到指定目录: {str(e)}"
        logger.error(error_msg)
        yield error_msg, None
        return

    # 准备文件路径
    file_name_without_ext = os.path.splitext(original_filename)[0]
    mp3_path = os.path.join(TEMP_DIR, f"{file_name_without_ext}.mp3")
    wav_path = os.path.join(TEMP_DIR, f"{file_name_without_ext}.wav")
    japanese_srt_path = os.path.join(OUTPUT_DIR, f"{file_name_without_ext}_jp.srt")
    chinese_srt_path = os.path.join(OUTPUT_DIR, f"{file_name_without_ext}_zh.srt")

    try:
        # 初始化步骤时间变量
        step1_duration = 0
        step2_duration = 0
        step3_duration = 0
        step4_duration = 0
        
        # 步骤1: 视频转MP3 - 实时反馈 + 时间记录
        step1_start = time.time()
        elapsed = time.time() - process_start_time
        
        yield f"⏰ 已运行 {elapsed:.1f}s\n🎬 步骤 1/4: 正在将视频转换为 MP3 格式...", None
        await asyncio.sleep(0.1)
        
        yield f"⏰ 已运行 {time.time() - process_start_time:.1f}s\n🎬 步骤 1/4: 正在将视频转换为 MP3 格式...\n🔄 正在提取音频流，请稍候...", None
        await asyncio.sleep(0.1)
        
        logger.info(f"开始视频转MP3: {current_video_path} -> {mp3_path}")
        
        try:
            # 异步执行转换
            mp3_conversion_success = await asyncio.to_thread(
                processing.video_to_mp3_with_progress, current_video_path, mp3_path
            )
            
            if not mp3_conversion_success:
                yield "❌ 错误：视频没有检测到音频流，无法提取字幕。请确认视频文件有音轨。", None
                return
            
            if not os.path.exists(mp3_path):
                yield "❌ 错误：MP3转换失败，文件未生成。", None
                return
            
            step1_duration = time.time() - step1_start
            elapsed = time.time() - process_start_time
            yield f"⏰ 已运行 {elapsed:.1f}s\n✅ 步骤 1/4 完成！MP3 文件生成成功 (耗时: {step1_duration:.1f}s)", None
            await asyncio.sleep(0.1)
            
        except Exception as e:
            yield f"❌ 步骤 1/4 失败：视频转MP3过程中出错 - {str(e)}", None
            return

        # 步骤2: MP3转WAV - 实时反馈 + 时间记录
        step2_start = time.time()
        elapsed = time.time() - process_start_time
        
        yield f"⏰ 已运行 {elapsed:.1f}s\n✅ 步骤 1/4 完成！\n🔄 步骤 2/4: 正在将 MP3 转换为 WAV 格式...", None
        await asyncio.sleep(0.1)
        
        yield f"⏰ 已运行 {time.time() - process_start_time:.1f}s\n✅ 步骤 1/4 完成！\n🔄 步骤 2/4: 正在将 MP3 转换为 WAV 格式...\n正在进行音频格式转换...", None
        await asyncio.sleep(0.1)
        
        logger.info(f"开始MP3转WAV: {mp3_path} -> {wav_path}")
        
        try:
            # 异步执行转换
            await asyncio.to_thread(processing.mp3_to_wav, mp3_path, wav_path)
            
            if not os.path.exists(wav_path):
                yield "❌ 错误：WAV转换失败。", None
                return
            
            step2_duration = time.time() - step2_start
            elapsed = time.time() - process_start_time
            yield f"⏰ 已运行 {elapsed:.1f}s\n✅ 步骤 2/4 完成！WAV 文件生成成功 (耗时: {step2_duration:.1f}s)", None
            await asyncio.sleep(0.1)
            
        except Exception as e:
            yield f"❌ 步骤 2/4 失败：MP3转WAV过程中出错 - {str(e)}", None
            return

        # 即时清理MP3文件
        await asyncio.to_thread(cleanup_file, mp3_path, "MP3文件")
        gc.collect()
        
        elapsed = time.time() - process_start_time
        yield f"⏰ 已运行 {elapsed:.1f}s\n✅ 步骤 2/4 完成！WAV 文件生成成功\n🧹 已清理临时MP3文件", None
        await asyncio.sleep(0.1)

        # 步骤3: ASR处理 - 实时反馈 + 时间记录
        step3_start = time.time()
        elapsed = time.time() - process_start_time
        
        yield f"⏰ 已运行 {elapsed:.1f}s\n✅ 步骤 2/4 完成！\n🎤 步骤 3/4: 正在执行语音识别 (ASR)...", None
        await asyncio.sleep(0.1)
        
        yield f"⏰ 已运行 {time.time() - process_start_time:.1f}s\n✅ 步骤 2/4 完成！\n🎤 步骤 3/4: 正在执行语音识别 (ASR)...\n🔄 正在识别音频中的日文语音并生成字幕...", None
        await asyncio.sleep(0.1)
        
        logger.info(f"开始ASR处理: {wav_path} -> {japanese_srt_path}")
        
        try:
            # 异步执行ASR，传入ASR服务地址
            await asyncio.to_thread(
                processing.perform_asr_and_generate_srt, 
                wav_path, 
                japanese_srt_path, 
                asr_base_url
            )
            
            if not os.path.exists(japanese_srt_path):
                yield "❌ 错误：ASR和日文字幕生成失败。", None
                return
            
            step3_duration = time.time() - step3_start
            elapsed = time.time() - process_start_time
            yield f"⏰ 已运行 {elapsed:.1f}s\n✅ 步骤 3/4 完成！日文字幕生成成功 (耗时: {step3_duration:.1f}s)", None
            await asyncio.sleep(0.1)
            
        except Exception as e:
            yield f"❌ 步骤 3/4 失败：ASR处理过程中出错 - {str(e)}", None
            return

        # 即时清理WAV文件
        await asyncio.to_thread(cleanup_file, wav_path, "WAV文件")
        gc.collect()
        
        elapsed = time.time() - process_start_time
        yield f"⏰ 已运行 {elapsed:.1f}s\n✅ 步骤 3/4 完成！日文字幕生成成功\n🧹 已清理临时WAV文件", None
        await asyncio.sleep(0.1)

        # 步骤4: 翻译为中文 - 实时反馈 + 时间记录
        step4_start = time.time()
        elapsed = time.time() - process_start_time
        
        yield f"⏰ 已运行 {elapsed:.1f}s\n✅ 步骤 3/4 完成！\n🌏 步骤 4/4: 正在翻译字幕...", None
        await asyncio.sleep(0.1)
        
        yield f"⏰ 已运行 {time.time() - process_start_time:.1f}s\n✅ 步骤 3/4 完成！\n🌏 步骤 4/4: 正在翻译字幕...\n🔄 正在将日文字幕智能翻译为中文...", None
        await asyncio.sleep(0.1)
        
        logger.info(f"开始翻译: {japanese_srt_path} -> {chinese_srt_path}")
        
        try:
            # 异步执行翻译，传入OpenAI配置
            await asyncio.to_thread(
                processing.translate_srt_to_chinese, 
                japanese_srt_path, 
                chinese_srt_path,
                openai_base_url,
                openai_api_key,
                openai_model
            )
            
            if not os.path.exists(chinese_srt_path):
                yield "❌ 错误：中文翻译失败。", None
                return
            
            step4_duration = time.time() - step4_start
            elapsed = time.time() - process_start_time
            yield f"⏰ 已运行 {elapsed:.1f}s\n✅ 步骤 4/4 完成！中文字幕翻译成功 (耗时: {step4_duration:.1f}s)", None
            await asyncio.sleep(0.1)
            
        except Exception as e:
            yield f"❌ 步骤 4/4 失败：翻译过程中出错 - {str(e)}", None
            return

        # 即时清理日文SRT文件
        await asyncio.to_thread(cleanup_file, japanese_srt_path, "日文SRT文件")
        gc.collect()
        
        elapsed = time.time() - process_start_time
        yield f"⏰ 已运行 {elapsed:.1f}s\n✅ 步骤 4/4 完成！中文字幕翻译成功\n🧹 已清理临时日文字幕文件", None
        await asyncio.sleep(0.1)
        
        # 完成处理 - 显示详细时间统计
        total_duration = time.time() - process_start_time
        end_datetime = datetime.now()
        
        logger.info("视频字幕处理完成")
        final_status = f"""🎉 所有步骤完成！

📊 处理时间统计：
⏱️ 总耗时: {total_duration:.1f}秒 ({total_duration/60:.1f}分钟)
⏰ 开始时间: {start_datetime.strftime('%H:%M:%S')}
⏰ 结束时间: {end_datetime.strftime('%H:%M:%S')}

📈 各步骤耗时详情：
• 步骤 1 (视频转MP3): {step1_duration:.1f}s
• 步骤 2 (MP3转WAV): {step2_duration:.1f}s  
• 步骤 3 (语音识别): {step3_duration:.1f}s
• 步骤 4 (智能翻译): {step4_duration:.1f}s

✅ 步骤 1/4: 视频转MP3 ✅
✅ 步骤 2/4: MP3转WAV ✅
✅ 步骤 3/4: 语音识别 ✅
✅ 步骤 4/4: 智能翻译 ✅

📥 请点击下方下载中文字幕文件"""
        
        # 记录到日志
        logger.info(f"处理完成 - 文件：{original_filename}, 总耗时：{total_duration:.1f}s")
        logger.info(f"步骤耗时 - MP3:{step1_duration:.1f}s, WAV:{step2_duration:.1f}s, ASR:{step3_duration:.1f}s, 翻译:{step4_duration:.1f}s")
        
        yield final_status, gr.File(value=chinese_srt_path)
        
    except Exception as e:
        # 捕获所有未处理的异常
        error_msg = f"❌ 处理失败: {str(e)}"
        logger.error(error_msg)
        yield error_msg, None
        return

    finally:
        # 最终清理
        logger.info("执行最终清理...")
        
        # 异步清理文件
        await asyncio.to_thread(cleanup_file, video_path_in_temp_dir, "视频副本")
        await asyncio.to_thread(cleanup_file, mp3_path, "MP3文件(备用)")
        await asyncio.to_thread(cleanup_file, wav_path, "WAV文件(备用)")
        await asyncio.to_thread(cleanup_file, japanese_srt_path, "日文SRT文件(备用)")
        
        # 清理分块文件
        await asyncio.to_thread(cleanup_chunk_files)
        
        # 最终垃圾回收
        gc.collect()
        logger.info("清理完成")


# 定义 Gradio 界面
with gr.Blocks(title="视频字幕自动生成与翻译", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎬 视频字幕自动生成与翻译 (前端配置版)")
    gr.Markdown(
        "上传视频文件，我们将自动生成日文字幕并翻译成中文。\n\n"
        f"**支持格式**: {', '.join(SUPPORTED_VIDEO_FORMATS)}\n"
        "**特性**: 异步处理 + 实时状态反馈 + **前端模型配置** + 详细时间统计显示"
    )

    with gr.Row():
        with gr.Column(scale=1):
            # 模型配置区域
            with gr.Group():
                gr.Markdown("### 🔧 模型配置")
                asr_base_url = gr.Textbox(
                    label="ASR服务地址", 
                    value=DEFAULT_ASR_BASE_URL,
                    placeholder="例: http://localhost:7863",
                    info="语音识别服务的完整地址"
                )
                openai_base_url = gr.Textbox(
                    label="OpenAI API地址", 
                    value=DEFAULT_OPENAI_BASE_URL,
                    placeholder="例: http://localhost:30000/v1",
                    info="OpenAI兼容API的完整地址（需包含/v1后缀）"
                )
                openai_api_key = gr.Textbox(
                    label="OpenAI API密钥", 
                    value=DEFAULT_OPENAI_API_KEY,
                    placeholder="sk-xxxxxxxxxx 或 your-api-key",
                    info="API认证密钥"
                )
                openai_model = gr.Textbox(
                    label="OpenAI模型名称", 
                    value=DEFAULT_OPENAI_MODEL,
                    placeholder="例: Qwen2.5-7b, gpt-3.5-turbo",
                    info="用于翻译的模型名称"
                )
            
            # 文件上传和处理
            with gr.Group():
                gr.Markdown("### 📁 文件处理")
                video_input = gr.File(
                    label="上传视频文件", 
                    file_types=SUPPORTED_VIDEO_FORMATS
                )
                process_button = gr.Button("🚀 开始处理", variant="primary", size="lg")
            
        with gr.Column(scale=1):
            status_text = gr.Textbox(
                label="📊 处理状态 (实时更新 + ⏰时间统计)", 
                value="等待上传文件...", 
                interactive=False,
                lines=12,  # 增加行数以显示更多信息
                max_lines=20
            )
            output_srt = gr.File(
                label="📥 下载翻译后的SRT字幕", 
                interactive=False
            )

    # 添加使用说明
    with gr.Accordion("ℹ️ 使用说明 & 技术特性", open=False):
        gr.Markdown("""
        ### 🔧 模型配置说明
        - **ASR服务地址**: 语音识别服务的地址，通常运行在本地或服务器上
        - **OpenAI API地址**: 用于翻译的AI服务地址，可以是OpenAI官方API或兼容的本地服务
        - **API密钥**: 用于认证的密钥，本地服务通常可以使用任意值如"your-api-key"
        - **模型名称**: 具体的模型名称，如Qwen2.5-7b、gpt-3.5-turbo等
        
        ### 🔄 处理流程 (异步实时反馈)
        1. **步骤 1/4: 视频转MP3** 
            - 📁 文件验证和复制
            - 🎬 提取音频流并转换格式
            - ⚡ 实时显示转换进度
        2. **步骤 2/4: MP3转WAV** 
            - 🔄 音频格式优化
            - 🧹 自动清理MP3临时文件
        3. **步骤 3/4: 语音识别 (ASR)** 
            - 🎤 AI语音识别处理
            - 📝 生成日文字幕文件
            - 🧹 自动清理WAV临时文件
        4. **步骤 4/4: 智能翻译** 
            - 🌏 日文到中文智能翻译
            - 🧹 自动清理日文字幕临时文件
            - 📥 输出最终中文字幕
        
        ### ⚡ 技术特性
        - **前端配置**: 所有模型配置都在界面上，无需修改代码
        - **异步处理**: 使用async/await提高并发性能
        - **实时反馈**: 每个子步骤都有即时状态更新
        - **⏰ 详细时间统计**: 显示总耗时、各步骤耗时、开始结束时间
        - **实时计时器**: 处理过程中实时显示已运行时间
        - **智能清理**: 处理过程中自动清理临时文件
        - **错误恢复**: 详细的错误信息和处理状态
        
        ### 📊 时间监控信息
        - ⏰ 实时显示当前已运行时间
        - 📈 每个步骤完成后显示该步骤耗时
        - 🕐 最终显示总处理时间和详细统计
        - 📅 记录开始和结束的具体时间点
        - 📊 各步骤耗时对比分析
        
        ### 💡 性能优势
        - 前端配置，方便部署和修改
        - 异步I/O操作，不阻塞界面
        - 实时状态更新，用户体验更好
        - 自动资源管理，服务器负载更低
        - 支持并发处理多个文件
        """)

    # 处理函数的包装器
    async def handle_process(video_file, asr_url, openai_url, api_key, model_name):
        """异步处理函数的包装器"""
        if video_file is None:
            yield "❌ 请先上传视频文件", None
            return
        
        # 调用异步生成器函数
        async for status, file_output in process_video_for_gradio(
            video_file, asr_url, openai_url, api_key, model_name
        ):
            yield status, file_output

    # 绑定处理函数
    process_button.click(
        fn=handle_process,
        inputs=[video_input, asr_base_url, openai_base_url, openai_api_key, openai_model],
        outputs=[status_text, output_srt],
        api_name="process_video"
    )


# 将 Gradio 界面挂载到 FastAPI 应用
app = gr.mount_gradio_app(app, demo, path="/gradio")


@app.get("/")
async def root():
    """根路径重定向到Gradio界面"""
    return {
        "message": "视频字幕生成服务正在运行 (前端配置版)", 
        "gradio_url": "/gradio",
        "api_docs": "/docs",
        "health_check": "/health",
        "features": ["异步处理", "实时反馈", "前端模型配置", "Gradio自动监控", "智能清理"]
    }


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "service": "video-subtitle-generator-frontend-config",
        "version": "frontend-config",
        "temp_dir": TEMP_DIR,
        "output_dir": OUTPUT_DIR,
        "supported_formats": SUPPORTED_VIDEO_FORMATS,
        "features": {
            "async_processing": True,
            "realtime_feedback": True,
            "frontend_config": True,
            "gradio_monitoring": True,
            "auto_cleanup": True
        }
    }


@app.post("/api/generate_subtitles_raw/")
async def generate_subtitles_raw(
    video_file: UploadFile = File(...),
    asr_base_url: str = DEFAULT_ASR_BASE_URL,
    openai_base_url: str = DEFAULT_OPENAI_BASE_URL, 
    openai_api_key: str = DEFAULT_OPENAI_API_KEY,
    openai_model: str = DEFAULT_OPENAI_MODEL
):
    """
    异步版本的 FastAPI 端点，用于非 Gradio 的 API 调用。
    支持前端传入模型配置参数。
    
    Args:
        video_file: 上传的视频文件
        asr_base_url: ASR服务地址
        openai_base_url: OpenAI API地址
        openai_api_key: OpenAI API密钥
        openai_model: OpenAI模型名称
        
    Returns:
        FileResponse: 生成的中文SRT文件
        
    Raises:
        HTTPException: 处理过程中的各种错误
    """
    # 验证文件
    if not validate_video_file(video_file.filename):
        raise HTTPException(
            status_code=400, 
            detail=f"不支持的视频格式。请上传: {', '.join(SUPPORTED_VIDEO_FORMATS)}"
        )

    # 安全处理文件名
    safe_filename = os.path.basename(video_file.filename)
    video_path = os.path.join(TEMP_DIR, safe_filename)
    
    # 初始化文件路径变量
    mp3_path = ""
    wav_path = ""
    japanese_srt_path = ""
    chinese_srt_path = ""

    try:
        # 异步保存上传的视频文件
        logger.info(f"开始处理API请求: {safe_filename}")
        with open(video_path, "wb") as buffer:
            # 异步读取文件内容
            content = await video_file.read()
            buffer.write(content)

        # 准备文件路径
        file_name_without_ext = os.path.splitext(safe_filename)[0]
        mp3_path = os.path.join(TEMP_DIR, f"{file_name_without_ext}.mp3")
        wav_path = os.path.join(TEMP_DIR, f"{file_name_without_ext}.wav")
        japanese_srt_path = os.path.join(OUTPUT_DIR, f"{file_name_without_ext}_jp.srt")
        chinese_srt_path = os.path.join(OUTPUT_DIR, f"{file_name_without_ext}_zh.srt")

        # 步骤1: 异步视频转MP3
        logger.info(f"API: 开始视频转MP3: {video_path} -> {mp3_path}")
        mp3_conversion_success = await asyncio.to_thread(
            processing.video_to_mp3_with_progress, video_path, mp3_path
        )
        
        if not mp3_conversion_success or not os.path.exists(mp3_path):
            raise HTTPException(status_code=500, detail="MP3转换失败或视频无音频流。")

        # 步骤2: 异步MP3转WAV
        logger.info(f"API: 开始MP3转WAV: {mp3_path} -> {wav_path}")
        await asyncio.to_thread(processing.mp3_to_wav, mp3_path, wav_path)
        if not os.path.exists(wav_path):
            raise HTTPException(status_code=500, detail="WAV转换失败。")
        
        # 异步清理MP3
        await asyncio.to_thread(cleanup_file, mp3_path, "MP3文件(API)")
        gc.collect()

        # 步骤3: 异步ASR处理
        logger.info(f"API: 开始ASR处理: {wav_path} -> {japanese_srt_path}")
        await asyncio.to_thread(
            processing.perform_asr_and_generate_srt, 
            wav_path, 
            japanese_srt_path, 
            asr_base_url
        )
        if not os.path.exists(japanese_srt_path):
            raise HTTPException(status_code=500, detail="ASR和日文字幕生成失败。")

        # 异步清理WAV
        await asyncio.to_thread(cleanup_file, wav_path, "WAV文件(API)")
        gc.collect()

        # 步骤4: 异步翻译为中文
        logger.info(f"API: 开始翻译: {japanese_srt_path} -> {chinese_srt_path}")
        await asyncio.to_thread(
            processing.translate_srt_to_chinese, 
            japanese_srt_path, 
            chinese_srt_path,
            openai_base_url,
            openai_api_key,
            openai_model
        )
        if not os.path.exists(chinese_srt_path):
            raise HTTPException(status_code=500, detail="中文翻译失败。")

        # 异步清理日文SRT
        await asyncio.to_thread(cleanup_file, japanese_srt_path, "日文SRT文件(API)")
        gc.collect()

        logger.info(f"API处理完成: {chinese_srt_path}")
        return FileResponse(
            chinese_srt_path, 
            media_type="application/x-subrip", 
            filename=os.path.basename(chinese_srt_path)
        )

    except HTTPException:
        # 直接抛出HTTP异常
        raise
    except Exception as e:
        # 包装其他异常
        error_msg = f"处理失败: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
        
    finally:
        # 异步最终清理
        logger.info("API: 执行最终清理...")
        
        # 清理主要文件
        await asyncio.to_thread(cleanup_file, video_path, "上传视频文件(API)")
        await asyncio.to_thread(cleanup_file, mp3_path, "MP3文件(API备用)")
        await asyncio.to_thread(cleanup_file, wav_path, "WAV文件(API备用)")
        await asyncio.to_thread(cleanup_file, japanese_srt_path, "日文SRT文件(API备用)")
        
        # 清理分块文件
        await asyncio.to_thread(cleanup_chunk_files)
        
        # 最终垃圾回收
        gc.collect()
        logger.info("API: 清理完成")


if __name__ == "__main__":
    import uvicorn
    
    logger.info("启动视频字幕生成服务 (前端配置版)...")
    logger.info(f"Gradio界面: http://0.0.0.0:7866/gradio")
    logger.info(f"API文档: http://0.0.0.0:7866/docs")
    logger.info(f"健康检查: http://0.0.0.0:7866/health")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=7866,
        log_level="info"
    )