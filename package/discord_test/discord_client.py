#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   discord_client.py
@Time    :   2023-11-16 10:00
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2023, Wu Xiaomin
@Desc    :   
"""

import discord

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)


@client.event
async def on_ready():
    print(f"We have logged in as {client.user}")


@client.event
async def on_message(message):
    if message.author == client.user:
        return
    print(message.content)
    if message.content.startswith("$hello"):
        await message.channel.send("Hello!")


if __name__ == "__main__":
    client.run(
        "MTEwMzkyNzI5MTE0ODUwOTIwNA.G0WuOq.FHMRQQVWEghSPfkh1fJcnwiiWNeVg0PLvWjg8w"
    )
