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
        'rate': '95%',       # Biraz yavaş
        'pitch': '-5%',      # Biraz alçak ton
        'volume': 'soft',    # Biraz kısık ses
        'emphasis': 'reduced'
    },
    'fear': {
        'rate': '98%',       # Hafif yavaş
        'pitch': '-2%',      # Hafif alçak ton
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
    }
}

def handle_punctuation(text):
    # Önce duraklamaları ekle
    text = re.sub(r'\.(\s|$)', r'<break time="500ms"/>\1', text)
    text = re.sub(r'!(\s|$)', r'<break time="400ms"/>\1', text)
    text = re.sub(r'\?(\s|$)', r'<break time="450ms"/>\1', text)
    text = re.sub(r',(\s|$)', r'<break time="200ms"/>\1', text)
    text = re.sub(r';(\s|$)', r'<break time="300ms"/>\1', text)
    text = re.sub(r':(\s|$)', r'<break time="250ms"/>\1', text)
    text = re.sub(r'\.{3}(\s|$)', r'<break time="700ms"/>\1', text)
    
    # Tüm noktalama işaretlerini temizle
    text = re.sub(r'[.,!?;:]', '', text)
    
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

def analyze_sentence_emotions(text):
    # Cümleleri ayır
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    # Her cümle için duygu analizi yap
    emotion_analyzer = pipeline("text-classification", 
                              model="j-hartmann/emotion-english-distilroberta-base", 
                              return_all_scores=True)
    
    sentence_emotions = []
    for sentence in sentences:
        results = emotion_analyzer(sentence)
        max_emotion = max(results[0], key=lambda x: x['score'])
        sentence_emotions.append({
            'text': sentence,
            'emotion': max_emotion['label'],
            'score': max_emotion['score']
        })
    
    return sentence_emotions

def get_contextual_ssml(sentence_emotions):
    ssml_parts = []
    
    for i, current in enumerate(sentence_emotions):
        # Önceki ve sonraki cümlelerin duygularını kontrol et
        prev_emotion = sentence_emotions[i-1] if i > 0 else None
        next_emotion = sentence_emotions[i+1] if i < len(sentence_emotions)-1 else None
        
        # Duygu değişimini kontrol et
        emotion_change = False
        
        # Önceki cümleyle karşılaştır
        if prev_emotion and prev_emotion['emotion'] != current['emotion']:
            # Keskin duygu değişimi var mı?
            if current['score'] > 0.8 and prev_emotion['score'] > 0.8:
                emotion_change = True
                
        # Sonraki cümleyle karşılaştır
        if next_emotion and next_emotion['emotion'] != current['emotion']:
            # Keskin duygu değişimi var mı?
            if current['score'] > 0.8 and next_emotion['score'] > 0.8:
                emotion_change = True
        
        # Duygu değişimi varsa veya güçlü bir duygu varsa
        if emotion_change or current['score'] > 0.8:
            # Duyguya göre SSML uygula
        
            settings = emotion_settings[current['emotion'].lower()]
            text = add_punctuation_effects(current['text'])
            ssml = f"""<prosody rate="{settings['rate']}" pitch="{settings['pitch']}" 
                      volume="{settings['volume']}">
                      <emphasis level="{settings['emphasis']}">
                          {text}
                      </emphasis>
                      </prosody>"""
        else:
            # Normal ton kullan
            text = add_punctuation_effects(current['text'])
            ssml = f"<prosody rate='100%' pitch='0%'>{text}</prosody>"
            
        ssml_parts.append(ssml)
    
    # Tüm SSML parçalarını birleştir
    final_ssml = f"""<speak>
        <break time="300ms"/>
        {' <break time="500ms"/> '.join(ssml_parts)}
        <break time="300ms"/>
    </speak>"""
    
    return final_ssml

def text_to_speech_and_emotion(text, output_file="output.mp3"):
    try:
        # Metni temizle
        text = text.strip()
        text = text.replace('&', 'and')
        text = ''.join(char for char in text if char.isalnum() or char in ' .,!?-')
        
        # Cümle bazlı duygu analizi
        sentence_emotions = analyze_sentence_emotions(text)
        
        # Her cümlenin duygusunu yazdır
        print("\nSentence emotions:")
        for i, emotion_data in enumerate(sentence_emotions, 1):
            print(f"Sentence {i} -> Emotion: {emotion_data['emotion']} ({emotion_data['score']:.2%})")
            print("-" * 50)
        
        # Bağlamsal SSML oluştur
        speech_text = get_contextual_ssml(sentence_emotions)
        
        # Dominant duyguyu bul ve yazdır
        dominant_emotion = max(sentence_emotions, key=lambda x: x['score'])
        print(f"\nDominant emotion: {dominant_emotion['emotion']} ({dominant_emotion['score']:.2%})")
        
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
