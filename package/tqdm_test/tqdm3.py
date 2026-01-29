import enlighten
import time
import sys


class MultiBarManager:
    def __init__(self):
        # 1. 初始化管理器
        self.manager = enlighten.get_manager()
        self.bars = {}
        self.counters = {}  # 存储具体的 counter 对象

        # 2. 建议：创建一个顶部状态栏，显示全局信息
        self.status_bar = self.manager.status_bar(
            status_format='Training System | Tasks: {tasks} | {time}',
            tasks=0,
            time=time.strftime('%H:%M:%S')
        )

    def add_task(self, task_id, total, desc):
        # enlighten 会自动处理行号，新任务会自动堆叠，不会产生阶梯空行
        pbar = self.manager.counter(
            total=total,
            desc=f"{desc:30}",  # 固定描述宽度，防止左右对齐抖动
            unit='f',
            color='cyan',
            leave=True
        )
        self.bars[task_id] = pbar
        # 更新状态栏
        self.status_bar.update(tasks=len(self.bars), time=time.strftime('%H:%M:%S'))
        return pbar

    def update_task(self, task_id, n=1, **kwargs):
        """
        update 传递数值，kwargs 传递想要在右侧显示的指标
        """
        if task_id in self.bars:
            pbar = self.bars[task_id]
            pbar.update(n)
            # 如果有额外的指标（如 queue_size），可以用此方式更新
            # 注意：enlighten 的 postfix 更新方式与 tqdm 略有不同
            # 它通常直接在 desc 或 format 中定义，或者简单地打印日志

    def log(self, message):
        """
        ✅ 关键修正：使用 write 替代 log
        这会将文字安全地插入到所有进度条上方
        """
        self.manager.write(message)

    def stop(self):
        self.manager.stop()


if __name__ == '__main__':
    mgr = MultiBarManager()

    # 模拟你的 Learner 列表
    learner_names = [
        "Learner-single-g0-1",
        "Learner-team-g0-1",
        "Learner-single-g1-1",
        "Learner-team-g1-1"
    ]

    for name in learner_names:
        mgr.add_task(name, 9090909090909, name)

    try:
        for i in range(1, 101):
            time.sleep(0.1)
            for name in learner_names:
                mgr.update_task(name, 100000000)

            if i % 10 == 0:
                mgr.log(f"[{time.strftime('%H:%M:%S')}] 系统信息: 批次 {i} 处理完成")

    except KeyboardInterrupt:
        pass
    finally:
        mgr.stop()