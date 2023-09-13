from aiogram import Bot, types
from aiogram import Dispatcher
from aiogram.filters import CommandStart
from aiogram.filters import Command
from aiogram import F
import asyncio
import logging
import json
import re
import random
from aiogram.utils.keyboard import ReplyKeyboardBuilder

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("horo_bot/models")
model = AutoModelForSeq2SeqLM.from_pretrained("horo_bot/models")

with open("horo_bot/config.json", "r") as f:
    json_config = json.load(f)
TOKEN = json_config["token"]

logging.basicConfig(level=logging.INFO)

bot = Bot(TOKEN)
dp = Dispatcher()

lst = [
    "♈Овен",
    "♌Лев",
    "♐Стрелец",
    "♋Рак",
    "♏Скорпион",
    "♓Рыбы",
    "♉Телец",
    "♍Дева",
    "♑Козерог",
    "♊Близнецы",
    "♎Весы",
    "♒Водолей",
]

dialog = {
    "in": [
        "/hello",
        "привет",
        "hello",
        "hi",
        "privet",
        "hey",
        "добрый день",
        "доброе утро",
        "добрый вечер",
    ],
    "out": ["Приветствую", "Здравствуйте", "Привет!", "Hi!", "Звездный привет!"],
}


# start message
@dp.message(CommandStart())
async def start(message: types.Message):
    kb = [
        [
            types.KeyboardButton(text="🔥Знаки Огня"),
            types.KeyboardButton(text="💧Знаки Воды"),
            types.KeyboardButton(text="🌍Знаки Земли"),
            types.KeyboardButton(text="🎈Знаки Воздуха"),
        ],
    ]

    start_menu = types.ReplyKeyboardMarkup(
        keyboard=kb, resize_keyboard=True, input_field_placeholder="Выбери свою стихию:"
    )

    await message.reply("Сперва выбери свою стихию:", reply_markup=start_menu)


# help message
@dp.message(Command("help"))
async def help(message: types.Message):
    kb = [
        [
            types.KeyboardButton(text="🔥Знаки Огня"),
            types.KeyboardButton(text="💧Знаки Воды"),
            types.KeyboardButton(text="🌍Знаки Земли"),
            types.KeyboardButton(text="🎈Знаки Воздуха"),
        ],
    ]

    start_menu = types.ReplyKeyboardMarkup(
        keyboard=kb, resize_keyboard=True, input_field_placeholder="Выбери свою стихию:"
    )
    await message.reply(
        """\
                     Привет, еще раз 😉!
Для начала работы выбери в меню свою стихию, а затем знак Зодиака. Напоминаю:\n\
- стихия Огня: ♈Овен, ♌Лев, ♐Стрелец\n\
- стихия Воды: ♋Рак, ♏Скорпион, ♓Рыбы\n\
- стихия Земли: ♉Телец, ♍Дева, ♑Козерог\n\
- стихия Воздуха: ♊Близнецы', ♎Весы, ♒Водолей""",
        reply_markup=start_menu,
    )


# any other text message
@dp.message(F.text)
async def handle_text(message: types.Message):
    if message.text == "🔥Знаки Огня":
        builder = ReplyKeyboardBuilder()
        builder.row(
            types.KeyboardButton(text="♈Овен"),
            types.KeyboardButton(text="♌Лев"),
            types.KeyboardButton(text="♐Стрелец"),
        )
        builder.row(types.KeyboardButton(text="Назад"))
        await message.reply(
            "Для какого знака Зодиака нужен гороскоп?",
            reply_markup=builder.as_markup(resize_keyboard=True),
        )

    elif message.text == "💧Знаки Воды":
        builder = ReplyKeyboardBuilder()
        builder.row(
            types.KeyboardButton(text="♋Рак"),
            types.KeyboardButton(text="♏Скорпион"),
            types.KeyboardButton(text="♓Рыбы"),
        )
        builder.row(types.KeyboardButton(text="Назад"))
        await message.reply(
            "Для какого знака Зодиака нужен гороскоп?",
            reply_markup=builder.as_markup(resize_keyboard=True),
        )

    elif message.text == "🌍Знаки Земли":
        builder = ReplyKeyboardBuilder()
        builder.row(
            types.KeyboardButton(text="♉Телец"),
            types.KeyboardButton(text="♍Дева"),
            types.KeyboardButton(text="♑Козерог"),
        )
        builder.row(types.KeyboardButton(text="Назад"))
        await message.reply(
            "Для какого знака Зодиака нужен гороскоп?",
            reply_markup=builder.as_markup(resize_keyboard=True),
        )

    elif message.text == "🎈Знаки Воздуха":
        builder = ReplyKeyboardBuilder()
        builder.row(
            types.KeyboardButton(text="♊Близнецы"),
            types.KeyboardButton(text="♎Весы"),
            types.KeyboardButton(text="♒Водолей"),
        )
        builder.row(types.KeyboardButton(text="Назад"))
        await message.reply(
            "Для какого знака Зодиака нужен гороскоп?",
            reply_markup=builder.as_markup(resize_keyboard=True),
        )

    elif message.text == "Назад":
        kb = [
            [
                types.KeyboardButton(text="🔥Знаки Огня"),
                types.KeyboardButton(text="💧Знаки Воды"),
                types.KeyboardButton(text="🌍Знаки Земли"),
                types.KeyboardButton(text="🎈Знаки Воздуха"),
            ],
        ]

        start_menu = types.ReplyKeyboardMarkup(
            keyboard=kb,
            resize_keyboard=True,
            input_field_placeholder="Выбери свою стихию:",
        )
        await message.reply("Сперва выбери свою стихию:", reply_markup=start_menu)

    elif any(x in message.text for x in lst):
        await message.answer(
            "{0.first_name}!\nСекунду, 👀смотрю на звезды...".format(message.from_user),
            reply_markup=types.ReplyKeyboardRemove(),
        )
        input_ids = tokenizer.encode(message.text, return_tensors="pt")
        sample_output = model.generate(
            input_ids,
            do_sample=True,
            max_length=210,
            top_p=0.7,
            temperature=0.6,
            top_k=0,
            no_repeat_ngram_size=2,
            eos_token_id=tokenizer.eos_token_id,
        )

        out = tokenizer.decode(sample_output[0][1:], skip_special_tokens=True)
        out = re.sub(re.compile("[^а-яА-ЯЁё !.,:?;-]"), "", out)
        symbols = [
            "..",
            ",.",
            "?.",
            " , ",
            " . ",
            " : ",
            ".:",
            ":.",
            "??",
            "?!",
            ".?",
            "? .",
            "? .",
            " .",
            ",,",
            "::",
            "  ",
            ".и",
            ".или",
            ".!",
            ".,",
        ]
        for symb in symbols:
            out = out.replace(symb, "")
        for symb in symbols:
            out = out.replace(symb, "")
        for symb in symbols:
            out = out.replace(symb, "")
        if "</s>" in out:
            out = out[: out.find("</s>")].strip()
        kb = [
            [
                types.KeyboardButton(text="🔥Знаки Огня"),
                types.KeyboardButton(text="💧Знаки Воды"),
                types.KeyboardButton(text="🌍Знаки Земли"),
                types.KeyboardButton(text="🎈Знаки Воздуха"),
            ],
        ]

        start_menu = types.ReplyKeyboardMarkup(
            keyboard=kb,
            resize_keyboard=True,
            input_field_placeholder="Выбери свою стихию:",
        )
        await message.answer(
            f"Гороскоп для знака {message.text}:\n\n{out}", reply_markup=start_menu
        )

    elif any(x in message.text.lower() for x in dialog["in"]):
        kb = [
            [
                types.KeyboardButton(text="🔥Знаки Огня"),
                types.KeyboardButton(text="💧Знаки Воды"),
                types.KeyboardButton(text="🌍Знаки Земли"),
                types.KeyboardButton(text="🎈Знаки Воздуха"),
            ],
        ]

        start_menu = types.ReplyKeyboardMarkup(
            keyboard=kb,
            resize_keyboard=True,
            input_field_placeholder="Выбери свою стихию:",
        )
        await message.answer(random.choice(dialog["out"]), reply_markup=start_menu)
    else:
        kb = [
            [
                types.KeyboardButton(text="🔥Знаки Огня"),
                types.KeyboardButton(text="💧Знаки Воды"),
                types.KeyboardButton(text="🌍Знаки Земли"),
                types.KeyboardButton(text="🎈Знаки Воздуха"),
            ],
        ]

        start_menu = types.ReplyKeyboardMarkup(
            keyboard=kb,
            resize_keyboard=True,
            input_field_placeholder="Выбери свою стихию:",
        )
        await message.reply(
            "Привет! Не могу понять твой запрос😅.Пожалуйста, воспользуйся меню для получения гороскопа😉!!!",
            reply_markup=start_menu,
        )


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
