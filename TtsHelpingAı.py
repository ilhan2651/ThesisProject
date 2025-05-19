from transformers import pipeline
import boto3
import os
import time
from datetime import datetime
import re

OUTPUT_DIR = "speech_files"  # Çıktı klasörü

# Global emotion settings
emotion_settings = {
    'neutral': {
        'rate': '100%',
        'pitch': '0%',
        'volume': 'medium',
        'emphasis': 'none'
    },
    'joy': {
        'rate': '115%',      # 105%'ten 115%'e (daha hızlı)
        'pitch': '+15%',     # +10%'dan +15%'e (daha yüksek ton)
        'volume': 'x-loud',  # loud'dan x-loud'a (en yüksek ses)
        'emphasis': 'strong'  # moderate'den strong'a (daha güçlü vurgu)
    },
    'sadness': {
        'rate': '85%',        # 95%'ten 85%'e (daha yavaş)
        'pitch': '-15%',      # -5%'ten -15%'e (daha alçak ton)
        'volume': 'soft',     # Kısık ses
        'emphasis': 'reduced' # Azaltılmış vurgu
    },
    'fear': {
        'rate': '108%',      # Biraz hızlı (panik etkisi)
        'pitch': '-5%',      # Hafif alçak ton
        'volume': 'medium',  # Normal ses
        'emphasis': 'moderate'
    },
    'surprise': {
        'rate': '108%',      # Hafif hızlı
        'pitch': '+15%',     # Yüksek ton
        'volume': 'loud',    # Yüksek ses
        'emphasis': 'strong'
    },
    'disgust': {
        'rate': '97%',       # Hafif yavaş
        'pitch': '-10%',     # Alçak ton
        'volume': 'medium',  # Normal ses
        'emphasis': 'strong'
    },
    'anger': {
        'rate': '110%',      # Kontrollü hızlı
        'pitch': '+8%',      # Hafif yüksek ton
        'volume': 'loud',    # Yüksek ses
        'emphasis': 'strong'
    }
}

# Duygu kelimeleri sözlüğü
emotion_words = {
    'joy': {
        'high_intensity': ['excellent', 'amazing', 'wonderful', 'fantastic', 'incredible', 'overjoyed', 'ecstatic'],
        'medium_intensity': ['happy', 'glad', 'pleased', 'delighted', 'cheerful', 'joyful', 'content'],
        'low_intensity': ['nice', 'good', 'fine', 'okay', 'satisfied', 'pleasant']
    },
    'sadness': {
        'high_intensity': ['devastated', 'heartbroken', 'miserable', 'depressed', 'grief-stricken'],
        'medium_intensity': ['sad', 'unhappy', 'disappointed', 'down', 'blue', 'gloomy'],
        'low_intensity': ['upset', 'displeased', 'dissatisfied', 'regretful']
    },
    'fear': {
        'high_intensity': ['terrified', 'horrified', 'panicked', 'petrified'],
        'medium_intensity': ['scared', 'afraid', 'fearful', 'anxious', 'worried'],
        'low_intensity': ['nervous', 'uneasy', 'concerned', 'apprehensive']
    },
    'surprise': {
        'high_intensity': ['shocked', 'astounded', 'astonished', 'stunned'],
        'medium_intensity': ['surprised', 'amazed', 'startled'],
        'low_intensity': ['unexpected', 'unusual', 'curious']
    },
    'disgust': {
        'high_intensity': ['revolted', 'disgusted', 'repulsed', 'sickened'],
        'medium_intensity': ['dislike', 'distaste', 'aversion'],
        'low_intensity': ['unpleasant', 'uncomfortable', 'disagreeable']
    }
}

# Özel duygu kelimeleri listesini güncelle
emotional_words = {
    'fear': [
        'terrified', 'scared', 'afraid', 'frightened', 'horrified', 
        'panicked', 'trembling', 'shaking', 'terror', 'horror',
        'dread', 'panic', 'threat', 'danger', 'scary', 'dark',
        'shadow', 'nightmare', 'scream', 'hide', 'run', 'escape'
    ],
    'anger': [
        'angry', 'furious', 'rage', 'outraged', 'mad', 'hate',
        'destroy', 'revenge', 'fight', 'enough', 'stop', 'never',
        'dare', 'wrong', 'fault', 'blame', 'unfair', 'betrayed',
        'frustrated', 'irritated', 'annoyed', 'hostile'
    ]
}

# Duygu geçiş ayarları
emotion_transition_settings = {
    ('joy', 'sadness'): {
        'rate_change': -0.15,  # Yavaşlama
        'pitch_change': -0.20,  # Ton düşüşü
        'transition_time': '800ms'
    },
    ('sadness', 'joy'): {
        'rate_change': 0.10,   # Yavaşça hızlanma
        'pitch_change': 0.15,  # Ton yükselişi
        'transition_time': '600ms'
    },
    ('neutral', 'joy'): {
        'rate_change': 0.05,
        'pitch_change': 0.10,
        'transition_time': '400ms'
    },
    # Diğer duygu geçişleri...
}

def handle_punctuation(text):
    # Sadece duraklamaları ekle, noktalama işaretlerini silme
    text = re.sub(r'\.(\s|$)', r'<break time="500ms"/>.', text)
    text = re.sub(r'!(\s|$)', r'<break time="400ms"/>!', text)
    text = re.sub(r'\?(\s|$)', r'<break time="450ms"/>?', text)
    text = re.sub(r',(\s|$)', r'<break time="200ms"/>,', text)
    text = re.sub(r';(\s|$)', r'<break time="300ms"/>;', text)
    text = re.sub(r':(\s|$)', r'<break time="250ms"/>:', text)
    text = re.sub(r'\.{3}(\s|$)', r'<break time="700ms"/>...', text)
    
    return text.strip()

def add_punctuation_effects(text):
    return handle_punctuation(text)

def extend_adjectives(text):
    # Basit kelime listesi kullan
    adjectives = ['good', 'great', 'happy', 'beautiful', 'amazing', 'wonderful', 'lovely', 'nice', 'excellent', 'fantastic']
    
    words = text.split()
    modified_words = []
    
    for word in words:
        word_lower = word.lower()
        if word_lower in adjectives:
            # Kelimenin ortasındaki sesliyi uzat
            vowels = 'aeiou'
            mid_point = len(word) // 2
            for i in range(mid_point-1, -1, -1):
                if word[i].lower() in vowels:
                    word = word[:i] + word[i]*3 + word[i+1:]
                    break
        modified_words.append(word)
    
    return ' '.join(modified_words)

def add_joy_emphasis(text):
    # Önce sıfatları uzat
    text = extend_adjectives(text)
    
    # Yaygın pozitif sıfatları ve ünlemleri daha güçlü vurgula
    joy_words = {
        'amazing': '<prosody rate="120%" pitch="+70%">amaaazing</prosody>',
        'awesome': '<prosody rate="120%" pitch="+70%">aweeeesome</prosody>',
        'great': '<prosody rate="120%" pitch="+65%">greeeat</prosody>',
        'wonderful': '<prosody rate="120%" pitch="+70%">woooonderful</prosody>',
        'happy': '<prosody rate="120%" pitch="+65%">haaappy</prosody>',
        'excited': '<prosody rate="125%" pitch="+70%">exciiited</prosody>',
        'love': '<prosody rate="120%" pitch="+65%">loooove</prosody>',
        'beautiful': '<prosody rate="120%" pitch="+65%">beautifuuul</prosody>',
        'perfect': '<prosody rate="120%" pitch="+70%">perfeeect</prosody>',
        'fantastic': '<prosody rate="125%" pitch="+70%">fantaaastic</prosody>'
    }
    
    # Ünlem işaretlerini daha enerjik yap
    text = text.replace('!', '<break strength="strong"/> <prosody pitch="+45%" rate="130%">!</prosody> <break time="200ms"/>')
    
    # Pozitif kelimeleri vurgula
    for word, emphasis in joy_words.items():
        text = text.replace(f' {word} ', f' {emphasis} ')
        text = text.replace(f' {word}!', f' {emphasis}!')
        text = text.replace(f' {word}.', f' {emphasis}.')
    
    return text

def get_emotion_ssml(text, emotion, score):
    emotion_settings = {
        'joy': {
            'rate': '125%',
            'pitch': '+30%',
            'volume': 'x-loud',
            'emphasis': 'strong'
        },
        'sadness': {
            'rate': '95%',
            'pitch': '-10%',
            'volume': 'soft',
            'emphasis': 'reduced'
        },
        'fear': {
            'rate': '105%',
            'pitch': '+5%',
            'volume': 'soft',
            'emphasis': 'moderate'
        },
        'surprise': {
            'rate': '120%',
            'pitch': '+35%',
            'volume': 'loud',
            'emphasis': 'strong'
        },
        'disgust': {
            'rate': '110%',
            'pitch': '-20%',
            'volume': 'loud',
            'emphasis': 'strong'
        },
        'neutral': {
            'rate': '100%',
            'pitch': '0%',
            'volume': 'medium',
            'emphasis': 'none'
        }
    }
    
    settings = emotion_settings.get(emotion.lower(), emotion_settings['neutral'])
    processed_text = add_punctuation_effects(text)
    
    ssml = f"""<speak>
        <break time="300ms"/>
        <prosody rate="{settings['rate']}" pitch="{settings['pitch']}" volume="{settings['volume']}">
            <emphasis level="{settings['emphasis']}">
                {processed_text}
            </emphasis>
        </prosody>
        <break time="300ms"/>
        <prosody rate="100%" pitch="0%">
            Detected emotion is: {emotion}
        </prosody>
    </speak>"""
    
    return ssml

def get_word_emotion_intensity(word, emotion_type):
    """Kelimenin duygu yoğunluğunu belirle"""
    word = word.lower()
    if emotion_type in emotion_words:
        if word in emotion_words[emotion_type]['high_intensity']:
            return 'high'
        elif word in emotion_words[emotion_type]['medium_intensity']:
            return 'medium'
        elif word in emotion_words[emotion_type]['low_intensity']:
            return 'low'
    return None

def calculate_transition_parameters(prev_emotion, current_emotion, word_intensity):
    """Duygu geçiş parametrelerini hesapla"""
    base_settings = emotion_transition_settings.get(
        (prev_emotion, current_emotion),
        {'rate_change': 0, 'pitch_change': 0, 'transition_time': '300ms'}
    )
    
    # Yoğunluğa göre ayarla
    intensity_multiplier = {
        'high': 1.2,
        'medium': 1.0,
        'low': 0.8
    }.get(word_intensity, 1.0)
    
    return {
        'rate_change': base_settings['rate_change'] * intensity_multiplier,
        'pitch_change': base_settings['pitch_change'] * intensity_multiplier,
        'transition_time': base_settings['transition_time']
    }

def analyze_sentence_emotions(text):
    """Geliştirilmiş cümle duygu analizi"""
    # Cümleleri nokta VE ünlem işaretlerine göre ayır
    sentences = [s.strip() for s in re.split('[.!?]', text) if s.strip()]
    
    emotion_analyzer = pipeline("text-classification", 
                              model="j-hartmann/emotion-english-distilroberta-base", 
                              return_all_scores=True)
    
    sentence_emotions = []
    prev_emotion = 'neutral'
    
    for sentence in sentences:
        words = sentence.split()
        word_emotions = []
        
        # Her kelime için duygu yoğunluğunu kontrol et
        for word in words:
            word_intensity = None
            word_emotion = None
            
            # Tüm duygular için kelime kontrolü
            for emotion in emotion_words.keys():
                intensity = get_word_emotion_intensity(word, emotion)
                if intensity:
                    word_emotion = emotion
                    word_intensity = intensity
                    break
            
            word_emotions.append({
                'word': word,
                'emotion': word_emotion,
                'intensity': word_intensity
            })
        
        # Cümlenin genel duygu analizi
        results = emotion_analyzer(sentence)
        max_emotion = max(results[0], key=lambda x: x['score'])
        
        # Geçiş parametrelerini hesapla
        transition = calculate_transition_parameters(
            prev_emotion,
            max_emotion['label'],
            'medium'  # Varsayılan yoğunluk
        )
        
        sentence_emotions.append({
            'text': sentence,
            'emotion': max_emotion['label'],
            'score': max_emotion['score'],
            'word_emotions': word_emotions,
            'transition': transition,
            'prev_emotion': prev_emotion
        })
        
        prev_emotion = max_emotion['label']
    
    return sentence_emotions

def determine_overall_emotion(sentence_emotions):
    """Metnin ana duygusunu belirle"""
    # Duygu sayaçları ve toplam skorları
    emotion_stats = {}
    
    # Her duygu için istatistikleri topla
    for emotion_data in sentence_emotions:
        emotion = emotion_data['emotion']
        score = emotion_data['score']
        
        if emotion not in emotion_stats:
            emotion_stats[emotion] = {
                'count': 0,
                'high_scores': 0,  # >0.8 olan skorların sayısı
                'total_score': 0,
                'max_score': 0
            }
        
        emotion_stats[emotion]['count'] += 1
        emotion_stats[emotion]['total_score'] += score
        emotion_stats[emotion]['max_score'] = max(emotion_stats[emotion]['max_score'], score)
        
        if score > 0.8:
            emotion_stats[emotion]['high_scores'] += 1
    
    # En iyi duyguyu belirle
    best_emotion = None
    best_score = 0
    
    for emotion, stats in emotion_stats.items():
        # Ağırlıklı skor hesapla
        weighted_score = (
            stats['high_scores'] * 3 +  # Yüksek skorlu cümle sayısı önemli
            stats['count'] * 2 +        # Toplam cümle sayısı da önemli
            stats['total_score'] / len(sentence_emotions)  # Ortalama skor
        )
        
        # Eğer bu duygu diğerlerinden daha baskınsa
        if weighted_score > best_score:
            best_score = weighted_score
            best_emotion = emotion
    
    # En yüksek skoru al
    max_score = emotion_stats[best_emotion]['max_score']
    
    return best_emotion, max_score

def get_contextual_ssml(sentence_emotions, overall_emotion):
    """Bağlamsal SSML oluşturma"""
    ssml_parts = []
    
    for i, current in enumerate(sentence_emotions):
        words = current['text'].split()
        words_ssml = []
        current_emotion = current['emotion'].lower()
        
        # Her kelimeyi kontrol et
        for word in words:
            word_lower = word.lower()
            
            # Sadece cümlenin duygusu fear ise fear kelimelerini kontrol et
            if current_emotion == 'fear' and word_lower in emotional_words['fear']:
                # Korku kelimesi için özel SSML
                words_ssml.append(
                    f"""<prosody pitch="-10%" rate="105%">
                        <emphasis level="moderate">{word}</emphasis>
                    </prosody>"""
                )
            
            # Sadece cümlenin duygusu anger ise anger kelimelerini kontrol et
            elif current_emotion == 'anger' and word_lower in emotional_words['anger']:
                # Öfke kelimesi için özel SSML
                words_ssml.append(
                    f"""<prosody pitch="+10%" volume="loud">
                        <emphasis level="strong">{word}</emphasis>
                    </prosody>"""
                )
            
            else:
                words_ssml.append(word)
        
        # Cümle ayarlarını belirle
        if current_emotion == 'fear':
            settings = {
                'rate': '108%',
                'pitch': '-5%',
                'volume': 'medium'
            }
        elif current_emotion == 'anger':
            settings = {
                'rate': '110%',
                'pitch': '+8%',
                'volume': 'loud'
            }
        else:
            settings = emotion_settings.get(current_emotion, emotion_settings['neutral'])
        
        # Cümle SSML'ini oluştur
        sentence_ssml = f"""<prosody rate="{settings['rate']}" pitch="{settings['pitch']}" volume="{settings['volume']}">
            {' '.join(words_ssml)}
        </prosody><break time="200ms"/>"""
        
        ssml_parts.append(sentence_ssml)
    
    final_ssml = f"""<speak>{' '.join(ssml_parts)}</speak>"""
    return final_ssml

def text_to_speech_and_emotion(text, output_file="output.mp3"):
    try:
        # Metni temizle (sadece tehlikeli karakterleri temizle)
        text = text.strip()
        text = text.replace('&', 'and')
        # Noktalama işaretlerini koruyarak sadece izin verilen karakterleri tut
        text = ''.join(char for char in text if char.isalnum() or char in ' .,!?-\'')
        
        # Cümle bazlı duygu analizi
        sentence_emotions = analyze_sentence_emotions(text)
        
        # Her cümlenin duygusunu yazdır
        print("\nSentence emotions:")
        for i, emotion_data in enumerate(sentence_emotions, 1):
            print(f"Sentence {i} -> Emotion: {emotion_data['emotion']} ({emotion_data['score']:.2%})")
            print("-" * 50)
        
        # Genel duyguyu belirle
        overall_emotion, overall_score = determine_overall_emotion(sentence_emotions)
        
        # Bağlamsal SSML oluştur
        speech_text = get_contextual_ssml(sentence_emotions, overall_emotion)
        
        # Çıktıyı yazdır
        if overall_emotion:
            print(f"\nOverall emotion: {overall_emotion} ({overall_score:.2%})")
            print(f"Speech will be generated with overall {overall_emotion} tone")
        else:
            print("\nEach sentence will be read with its own emotion")
        
        # Polly istemcisini başlat
        polly_client = boto3.Session(
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
            region_name='eu-central-1'
        ).client('polly')
        
        # Sesi oluştur
        response = polly_client.synthesize_speech(
            Text=speech_text,
            TextType='ssml',
            OutputFormat='mp3',
            VoiceId='Kimberly'
        )
        
        # Ses dosyasını kaydet
        if 'AudioStream' in response:
            with open(output_file, 'wb') as file:
                file.write(response['AudioStream'].read())
        
        print(f"Audio file created")
        
        time.sleep(0.1)
        os.system(f"start {output_file}")
            
        return sentence_emotions
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def main():
    while True:
        print("\n" + "="*50)
        text = input("Please enter your text (Enter 'q' to quit): ")
        
        if text.lower() == 'q':
            print("Program terminating...")
            break
            
        text_to_speech_and_emotion(text)

if __name__ == "__main__":
    main()
