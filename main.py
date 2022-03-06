import discord
from discord.ext import commands

from chat_bot import *


def get_token_from_file(file_path: str):
    file = open(file_path, 'r')
    return file.read()


# date of creation 06 March 2022 | 13:14
training_mode = True
TOKEN = get_token_from_file("private/TOKEN.txt")
COMMAND_PREFIX = "-"
BOT_PREFIX = (COMMAND_PREFIX, "#bot.")

bot = commands.Bot(command_prefix=BOT_PREFIX, help_command=None)

cb = ChatBot(max_message_len=150, message_list_len=4)
cb.load_network_from_file()


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")


async def get_answer(channel):
    print(f"\033[94m> ({cb.get_answer()})\033[0m")
    expected_output = input("> ")

    cb.epoch(expected_output)
    cb.push_message("April: " + expected_output)
    await channel.send(expected_output)


@bot.event
async def on_message(message: discord.message.Message):
    if not message.author.bot:
        print("\n" * 100)
        print("\033[31mNew Message!\033[0m")
        new_message = message.author.name + ": " + message.content
        cb.push_message(new_message)
        if training_mode:
            for m in cb.last_messages:
                print(f"\033[90m# {''.join(m)}\033[0m")

            await get_answer(message.channel)
        else:
            for m in cb.last_messages:
                print(f"\033[90m# {''.join(m)}\033[0m")

            answer = cb.get_answer()
            await message.channel.send(answer)
            cb.push_message("april: " + answer)

            print("\n" * 100)
            for m in cb.last_messages:
                print(f"\033[90m# {''.join(m)}\033[0m")


if __name__ == '__main__':
    # cb.train(10000)
    bot.run(TOKEN)
