import discord
from discord.ext import commands

from chat_bot import *


def get_token_from_file(file_path: str):
    file = open(file_path, 'r')
    return file.read()


TOKEN = get_token_from_file("private/TOKEN.txt")
COMMAND_PREFIX = "-"
BOT_PREFIX = (COMMAND_PREFIX, "#bot.")

bot = commands.Bot(command_prefix=BOT_PREFIX, help_command=None)

cb = ChatBot(max_message_len=150, message_list_len=4)
cb.load_network_from_file()


# cb.train(1000)


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")


async def get_answer(channel):
    for m in cb.last_messages:
        print(f"\033[90m# {m}\033[0m")

    print(f"\033[94m> ({cb.get_answer()})\033[0m")
    expected_output = input("> ")

    cb.epoch(expected_output)
    cb.push_message(expected_output)
    await channel.send(expected_output)


@bot.event
async def on_message(message: discord.message.Message):
    if not message.author.bot:
        print("\n" * 100)
        new_message = message.content
        cb.push_message(new_message)

        await get_answer(message.channel)


bot.run(TOKEN)
"""
while True:
    some_input = input("Enter some input: ")
    cb.push_message(some_input)

    expected_answer = cb.epoch()
    cb.push_message(expected_answer)
    print()
"""
