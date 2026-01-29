from tqdm import tqdm
import time


class MultiBarManager:
    def __init__(self):
        self.bars = {}
        self.counter = 0

    def add_task(self, task_id, total, desc):
        pbar = tqdm(
            total=total,
            desc=desc,
            position=self.counter,
            leave=True,
            mininterval=999999,
            maxinterval=999999,
            miniters=999999
        )
        self.bars[task_id] = pbar
        self.counter += 1
        return pbar

    def set_desc(self, task_id, desc):
        """设置左侧描述"""
        if task_id in self.bars:
            # refresh=False 因为我们统一在主循环里刷新
            self.bars[task_id].set_description(desc, refresh=False)

    def set_post(self, task_id, **kwargs):
        """设置右侧指标 (支持字典传参)"""
        if task_id in self.bars:
            # refresh=False 避免此时触发输出
            self.bars[task_id].set_postfix(kwargs, refresh=False)

    def update_task(self, task_id, n=1):
        if task_id in self.bars:
            self.bars[task_id].update(n)

    def refresh_all(self):
        for pbar in self.bars.values():
            pbar.refresh()

    def close_all(self):
        self.refresh_all()
        for pbar in self.bars.values():
            pbar.close()
        print('\n' * self.counter)


# --- 使用演示 ---
if __name__ == '__main__':
    manager = MultiBarManager()
    manager.add_task("train", 100, "模型训练")

    for i in range(1, 101):
        time.sleep(0.05)
        manager.update_task("train", 1)

        # 动态修改左侧描述
        if i == 50:
            manager.set_desc("train", "⚡ 训练过半")

        # 动态修改右侧指标
        # 传入任意 key-value 都会显示在右边
        manager.set_post("train", loss=0.5 / i, acc=f"{i}%")

        manager.refresh_all()

    manager.close_all()