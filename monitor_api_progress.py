"""
API Progress Monitor - 监控LLM Embedding预计算进度
实时检测API调用速度是否正常
"""
import os
import re
import time
import sys
import subprocess
from datetime import datetime, timedelta

# 设置UTF-8编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def get_embedding_progress():
    """获取当前embedding预计算进度"""
    # 检查输出文件是否存在且正在增长
    embedding_file = "data/mind_small/llm_embeddings.npy"

    progress_info = {
        'file_exists': False,
        'file_size': 0,
        'completed': 0,
        'total': 51282,
        'batches_done': 0,
        'total_batches': 513
    }

    if os.path.exists(embedding_file):
        progress_info['file_exists'] = True
        progress_info['file_size'] = os.path.getsize(embedding_file) / (1024 * 1024)  # MB

    return progress_info


def parse_log_file():
    """尝试从日志中解析进度（如果有的话）"""
    # 如果有日志文件，可以从这里读取
    pass


def estimate_completion_time(batches_done, total_batches, avg_time_per_batch):
    """估算完成时间"""
    remaining_batches = total_batches - batches_done
    remaining_seconds = remaining_batches * avg_time_per_batch

    eta = datetime.now() + timedelta(seconds=remaining_seconds)
    return eta, remaining_seconds


def format_time(seconds):
    """格式化时间显示"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def check_api_speed():
    """检查API速度是否正常"""
    print("=" * 70)
    print("  LLM Embedding API 速度监控")
    print("=" * 70)
    print()

    # 正常速度参考
    NORMAL_SPEED = {
        'min_batches_per_min': 6,   # 最慢：10秒/批次
        'ideal_batches_per_min': 12,  # 理想：5秒/批次
        'max_batches_per_min': 20     # 最快：3秒/批次
    }

    print("📋 正常速度参考：")
    print(f"  理想速度：{NORMAL_SPEED['ideal_batches_per_min']} 批次/分钟 (~5秒/批次)")
    print(f"  可接受：{NORMAL_SPEED['min_batches_per_min']}-{NORMAL_SPEED['max_batches_per_min']} 批次/分钟")
    print()

    # 检查文件进度
    progress = get_embedding_progress()

    print("📂 输出文件状态：")
    if progress['file_exists']:
        print(f"  ✅ 文件存在：data/mind_small/llm_embeddings.npy")
        print(f"  📦 文件大小：{progress['file_size']:.2f} MB")
    else:
        print(f"  ❌ 文件不存在（可能还未开始写入）")
    print()

    # 进行速度测试：监控5分钟
    print("🔍 开始速度测试（监控5分钟）...")
    print("=" * 70)
    print()

    test_duration = 300  # 5分钟
    check_interval = 30  # 每30秒检查一次

    measurements = []
    start_time = time.time()

    # 获取初始状态
    if progress['file_exists']:
        initial_size = progress['file_size']
    else:
        initial_size = 0

    print(f"⏱️  开始时间：{datetime.now().strftime('%H:%M:%S')}")
    print(f"📊 初始文件大小：{initial_size:.2f} MB")
    print()

    for i in range(test_duration // check_interval):
        time.sleep(check_interval)

        elapsed = time.time() - start_time
        current_progress = get_embedding_progress()

        if current_progress['file_exists']:
            current_size = current_progress['file_size']
            growth = current_size - initial_size
            growth_rate = growth / (elapsed / 60)  # MB per minute

            # 预估批次速度（假设每100条新闻约12MB）
            estimated_batches = (growth / 12) * 100
            batches_per_min = estimated_batches / (elapsed / 60) if elapsed > 0 else 0

            measurements.append({
                'elapsed': elapsed,
                'size': current_size,
                'growth': growth,
                'batches_per_min': batches_per_min
            })

            # 显示进度
            print(f"⏱️  {format_time(elapsed)} | "
                  f"文件: {current_size:.2f} MB | "
                  f"增长: {growth:.2f} MB | "
                  f"速度: {batches_per_min:.1f} 批次/分钟")

            # 判断速度
            if batches_per_min < NORMAL_SPEED['min_batches_per_min']:
                status = "⚠️  慢"
            elif batches_per_min > NORMAL_SPEED['max_batches_per_min']:
                status = "⚡ 快"
            else:
                status = "✅ 正常"

            print(f"    状态: {status}")
            print()

    # 生成报告
    print()
    print("=" * 70)
    print("  速度测试报告")
    print("=" * 70)
    print()

    if measurements:
        avg_batches_per_min = sum(m['batches_per_min'] for m in measurements) / len(measurements)
        total_growth = measurements[-1]['growth']

        print(f"📊 测试时长：{format_time(measurements[-1]['elapsed'])}")
        print(f"📈 文件增长：{total_growth:.2f} MB")
        print(f"⚡ 平均速度：{avg_batches_per_min:.1f} 批次/分钟")
        print()

        # 判断
        if avg_batches_per_min < NORMAL_SPEED['min_batches_per_min']:
            print("❌ 速度异常慢！")
            print("   可能原因：")
            print("   1. API 限流")
            print("   2. 网络连接不稳定")
            print("   3. API 服务器负载高")
            print()
            print("   建议：")
            print("   - 检查网络连接")
            print("   - 稍后重试")
            print("   - 考虑减小 batch_size")
        elif avg_batches_per_min >= NORMAL_SPEED['min_batches_per_min'] and \
             avg_batches_per_min <= NORMAL_SPEED['max_batches_per_min']:
            print("✅ 速度正常！")

            # 预估完成时间
            if progress['file_exists']:
                # 假设总文件大小约 300MB (51282条 * 1536维 * 4字节)
                expected_total_size = 300
                remaining_size = expected_total_size - measurements[-1]['size']
                remaining_minutes = remaining_size / (total_growth / (measurements[-1]['elapsed'] / 60))

                eta = datetime.now() + timedelta(minutes=remaining_minutes)
                print(f"⏰ 预计完成时间：{eta.strftime('%H:%M:%S')}")
                print(f"⏳ 剩余时间：约 {format_time(remaining_minutes * 60)}")
        else:
            print("⚡ 速度非常快！")
            print("   API 调用顺畅，预计很快完成")
    else:
        print("⚠️  无法测量速度（文件可能未开始写入）")
        print()
        print("建议：")
        print("  1. 检查预计算进程是否正在运行")
        print("  2. 检查是否有错误日志")
        print("  3. 确认 API key 有效")

    print()
    print("=" * 70)


def simple_monitor():
    """简单监控模式 - 每分钟汇报一次"""
    print("=" * 70)
    print("  简单监控模式（每分钟更新）")
    print("  按 Ctrl+C 停止")
    print("=" * 70)
    print()

    last_size = 0
    last_time = time.time()

    try:
        while True:
            progress = get_embedding_progress()
            current_time = time.time()

            if progress['file_exists']:
                current_size = progress['file_size']
                elapsed = current_time - last_time

                if last_size > 0:
                    growth = current_size - last_size
                    growth_rate = growth / (elapsed / 60)  # MB/min

                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"文件大小: {current_size:.2f} MB | "
                          f"增长速度: {growth_rate:.2f} MB/min")
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"文件大小: {current_size:.2f} MB")

                last_size = current_size
                last_time = current_time
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 等待文件创建...")

            time.sleep(60)  # 每分钟检查一次

    except KeyboardInterrupt:
        print("\n\n监控已停止")


if __name__ == '__main__':
    import sys

    print("""
╔═══════════════════════════════════════════════════════════════╗
║         LLM Embedding API 速度监控工具                         ║
╚═══════════════════════════════════════════════════════════════╝
""")

    if len(sys.argv) > 1 and sys.argv[1] == '--simple':
        # 简单监控模式
        simple_monitor()
    else:
        # 速度测试模式
        check_api_speed()
