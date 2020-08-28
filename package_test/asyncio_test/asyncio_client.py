import time
import asyncio


async def hello():
    time.sleep(1)
    print("hello workld, {}".format(time.time()))


def run():
    for i in range(5):
        # loop.run_until_complete(hello())
        asyncio.run(hello())


def sync_hello():
    time.sleep(1)


def sync_run():
    for i in range(5):
        sync_hello()
        print('Hello World:%s' % time.time())  # 任何伟大的代码都是从Hello World 开始的！


loop = asyncio.get_event_loop()

if __name__ == '__main__':
    run()

    print("================================")
    sync_run()
