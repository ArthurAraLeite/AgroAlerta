import asyncio
from telegram import Bot
from config import TOKEN

CHAT_ID = "7858612258"

async def enviar_mensagem(chat_id: str, mensagem: str):
    bot = Bot(token=TOKEN)
    await bot.send_message(chat_id=chat_id, text=mensagem)

async def main():
    await enviar_mensagem(CHAT_ID, "Brote funcionando!")

asyncio.run(main())