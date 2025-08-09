# caption_styles.py

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

class CaptionGenerator:
    """Handles the creation of professional video captions in multiple styles."""
    def __init__(self, message_queue):
        self.message_queue = message_queue
        self.TARGET_RESOLUTION = (1080, 1920)
        
        # Le chemin de FONT_FOLDER est maintenant relatif au dossier où 'app.py' est exécuté.
        # En supposant que 'static' et 'caption_styles.py' sont dans le même répertoire racine du projet web.
        self.FONT_FOLDER = Path(__file__).parent / "static" / "fonts"

        self.KARAOKE_HIGHLIGHT_COLOR = '#FFFF00'
        self.KARAOKE_FUTURE_COLOR = '#FFFFFF'
        self.KARAOKE_PAST_COLOR = '#AAAAAA'
        self.DUAL_HIGHLIGHT_COLOR = '#00FF00'
        self.DUAL_BASE_COLOR = '#FFFFFF'
        self.FILLER_WORDS = {'A', 'AN', 'THE', 'TO', 'IS', 'ARE', 'OF', 'IN', 'ON', 'FOR', 'IT', 'WITH', 'BY', 'AT', 'FROM'}
        self.OPUS_HIGHLIGHT_COLOR = '#FFFFFF'
        self.OPUS_FUTURE_COLOR = "#32D316"
        self.OPUS_BLUE_COLOR = '#FFFFFF'
        self.OPUS_BLUE_HIGHLIGHT_COLOR = '#1676D3'

        self.REGULAR_FONT_SIZE = 80
        self.HIGHLIGHT_FONT_SIZE = 90

        self.FONTS = {
            "Gabarito": self.FONT_FOLDER / "Gabarito-Regular.ttf",
            "Anton": self.FONT_FOLDER / "Anton-Regular.ttf",
            "Montserrat": self.FONT_FOLDER / "montserrat-black-italic.ttf",
        }
        for font_name, font_path in self.FONTS.items():
            if not font_path.exists():
                self.message_queue.put(("log", f"⚠️ Font file not found: {font_path} for {font_name}."))

    def _create_image(self, lines_of_words, highlight_word_index, font_name, style_name):
        canvas = Image.new("RGBA", self.TARGET_RESOLUTION, (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        
        font_name_for_image = font_name
        if style_name in ["Opus Green/White", "Opus Blue/White"]:
            font_name_for_image = "Montserrat"
        
        font_path = self.FONTS.get(font_name_for_image, self.FONTS["Gabarito"])

        try:
            font_size_regular = 70 if font_name_for_image == "Montserrat" else self.REGULAR_FONT_SIZE
            font_size_highlight = 90 if font_name_for_image == "Montserrat" else self.HIGHLIGHT_FONT_SIZE
            font = ImageFont.truetype(str(font_path), font_size_regular)
            highlight_font = ImageFont.truetype(str(font_path), font_size_highlight)
        except IOError as e:
            self.message_queue.put(("log", f"⚠️ Font {font_name_for_image} at {font_path} not found. Using default. Error: {e}"))
            default_path = self.FONTS["Gabarito"]
            font = ImageFont.truetype(str(default_path), self.REGULAR_FONT_SIZE)
            highlight_font = ImageFont.truetype(str(default_path), self.HIGHLIGHT_FONT_SIZE)

        if style_name in ["Opus Green/White", "Opus Blue/White"]:
            lines_of_words = [[word.upper() for word in line] for line in lines_of_words]

        line_heights = [highlight_font.getbbox('A')[3]] * len(lines_of_words)
        total_text_height = sum(line_heights) + (15 * (len(lines_of_words) - 1))
        
        y_multiplier = 0.8 if style_name in ["Opus Green/White", "Opus Blue/White"] else 0.75
        current_y = (self.TARGET_RESOLUTION[1] * y_multiplier) - (total_text_height / 2)

        word_counter = 0
        for line_words in lines_of_words:
            line_width = 0
            for i, word in enumerate(line_words):
                is_highlight = (word_counter + i) == highlight_word_index
                current_font = highlight_font if is_highlight else font
                line_width += current_font.getlength(word)
            line_width += font.getlength(' ') * (len(line_words) - 1)
            
            current_x = (self.TARGET_RESOLUTION[0] - line_width) / 2
            
            for word in line_words:
                is_current_word = word_counter == highlight_word_index
                word_font = font
                
                if style_name == "Opus Green/White":
                    word_font = highlight_font if is_current_word else font
                    color = self.OPUS_FUTURE_COLOR if is_current_word else self.OPUS_HIGHLIGHT_COLOR
                elif style_name == "Opus Blue/White":
                    word_font = highlight_font if is_current_word else font
                    color = self.OPUS_BLUE_HIGHLIGHT_COLOR if is_current_word else self.OPUS_BLUE_COLOR
                elif style_name == "Dual Color":
                    color = self.DUAL_HIGHLIGHT_COLOR if word in self.FILLER_WORDS else self.DUAL_BASE_COLOR
                else: # Karaoke Style
                    if word_counter < highlight_word_index: color = self.KARAOKE_PAST_COLOR
                    elif is_current_word:
                        color = self.KARAOKE_HIGHLIGHT_COLOR
                        word_font = highlight_font
                    else: color = self.KARAOKE_FUTURE_COLOR

                draw.text((current_x, current_y), word, font=word_font, fill=color, stroke_width=8, stroke_fill="black")
                current_x += word_font.getlength(word) + font.getlength(' ')
                word_counter += 1

            current_y += highlight_font.getbbox('A')[3] + 15
        return canvas

    def create_caption_images(self, timed_words, font_name, style_name, temp_dir):
        output_images = []
        if not timed_words: return output_images

        self.message_queue.put((f"log", f"     🎨 Generating '{style_name}' caption images..."))
        
        max_words_per_group = 3 if style_name in ["Opus Green/White", "Opus Blue/White"] else 3
        word_groups = [timed_words[i:i + max_words_per_group] for i in range(0, len(timed_words), max_words_per_group)]
        
        img_idx = 0
        for group in word_groups:
            group_text_list = [w['text'] for w in group]
            
            lines_of_words = []
            if style_name in ["Opus Green/White", "Opus Blue/White"]:
                if len(group_text_list) == 3:
                    lines_of_words.append(group_text_list[0:2])
                    lines_of_words.append([group_text_list[2]])
                elif group_text_list:
                    lines_of_words.append(group_text_list)
            else: 
                if len(group_text_list) == 3:
                    lines_of_words.append(group_text_list[0:2])
                    lines_of_words.append([group_text_list[2]])
                else:
                    lines_of_words.append(group_text_list)

            for i, word in enumerate(group):
                pil_image = self._create_image(lines_of_words, i, font_name, style_name)
                clip_duration = word['end'] - word['start']
                if clip_duration > 0:
                    img_path = temp_dir / f"caption_{img_idx:04d}.png"
                    pil_image.save(img_path)
                    output_images.append({
                        "path": img_path,
                        "start": word['start'],
                        "end": word['end']
                    })
                    img_idx += 1
        
        self.message_queue.put((f"log", f"     ✅ Generated {len(output_images)} caption images."))
        return output_images