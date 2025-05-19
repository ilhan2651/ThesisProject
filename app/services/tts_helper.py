from transformers import pipeline
import boto3
import os
import time
import re

emotion_settings = {
    'neutral': {'rate': '100%', 'pitch': '0%', 'volume': 'medium', 'emphasis': 'none'},
    'joy': {'rate': '110%', 'pitch': '+20%', 'volume': 'x-loud', 'emphasis': 'strong'},
    'sadness': {'rate': '98%', 'pitch': '-15%', 'volume': 'soft', 'emphasis': 'reduced'},
    'fear': {'rate': '108%', 'pitch': '-5%', 'volume': 'medium', 'emphasis': 'moderate'},
    'surprise': {'rate': '108%', 'pitch': '+15%', 'volume': 'loud', 'emphasis': 'strong'},
    'disgust': {'rate': '95%', 'pitch': '-15%', 'volume': 'medium', 'emphasis': 'strong'},
    'anger': {'rate': '110%', 'pitch': '-15%', 'volume': 'loud', 'emphasis': 'none'}
}

emotion_keywords = {
    'joy': ['happy', 'joyful', 'delighted', 'excited', 'wonderful', 'fantastic', 'amazing', 'great', 'beautiful', 'perfect', 'love', 'awesome'],
    'sadness': ['sad', 'miserable', 'heartbroken', 'depressed', 'gloomy', 'sorrowful', 'unhappy', 'melancholy', 'grief', 'tearful'],
    'fear': ['afraid', 'scared', 'terrified', 'frightened', 'anxious', 'horrified', 'dreadful', 'fearful', 'nervous', 'worried'],
    'surprise': ['surprised', 'astonished', 'amazed', 'stunned', 'shocked', 'incredible', 'unexpected', 'remarkable', 'unbelievable', 'astounding'],
    'disgust': ['disgusting', 'revolting', 'repulsive', 'nauseating', 'vile', 'horrible', 'awful', 'terrible', 'gross', 'sickening'],
    'anger': ['angry', 'furious', 'enraged', 'outraged', 'irritated', 'annoyed', 'frustrated', 'fuming', 'livid', 'infuriated']
}

emotion_transition_settings = {
    ('joy', 'sadness'): {'rate_change': -0.15, 'pitch_change': -0.20, 'transition_time': '800ms'},
    ('sadness', 'joy'): {'rate_change': 0.10, 'pitch_change': 0.15, 'transition_time': '600ms'},
    ('neutral', 'joy'): {'rate_change': 0.05, 'pitch_change': 0.10, 'transition_time': '400ms'}
}

def handle_punctuation(text):
    text = re.sub(r'\.(\s|$)', r'<break time="500ms"/>.', text)
    text = re.sub(r'!(\s|$)', r'<break time="400ms"/>!', text)
    text = re.sub(r'\?(\s|$)', r'<break time="450ms"/>?', text)
    text = re.sub(r',(\s|$)', r'<break time="200ms"/>,', text)
    text = re.sub(r';(\s|$)', r'<break time="300ms"/>;', text)
    text = re.sub(r':(\s|$)', r'<break time="250ms"/>:', text)
    text = re.sub(r'\.{3}(\s|$)', r'<break time="700ms"/>...', text)
    return text.strip()

def add_punctuation_effects(text): return handle_punctuation(text)

def extend_adjectives(text):
    # Kelimeleri uzatma kuralları
    extend_rules = {
        'amazing': 'amaaazing',
        'awesome': 'aweeeesome',
        'great': 'greeeat',
        'wonderful': 'woooonderful',
        'happy': 'haaappy',
        'excited': 'exciiited',
        'love': 'loooove',
        'beautiful': 'beautifuuul',
        'perfect': 'perfeeect',
        'fantastic': 'fantaaastic'
    }
    
    # Her kelimeyi kontrol et ve uzat
    words = text.split()
    for i, word in enumerate(words):
        word_lower = word.lower().strip(".,!?;:")
        if word_lower in extend_rules:
            words[i] = extend_rules[word_lower]
    
    return ' '.join(words)

def add_joy_emphasis(text):
    text = extend_adjectives(text)
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
    text = text.replace('!', '<break strength="strong"/> <prosody pitch="+45%" rate="130%">!</prosody> <break time="200ms"/>')
    for word, emphasis in joy_words.items():
        text = text.replace(f' {word} ', f' {emphasis} ')
        text = text.replace(f' {word}!', f' {emphasis}!')
        text = text.replace(f' {word}.', f' {emphasis}.')
    return text

def get_emotion_ssml(text, emotion, score):
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
    word = word.lower()
    if emotion_type in emotion_keywords:
        if word in emotion_keywords[emotion_type]: return 'high'
    return None

def calculate_transition_parameters(prev_emotion, current_emotion, word_intensity):
    base = emotion_transition_settings.get((prev_emotion, current_emotion), {'rate_change': 0, 'pitch_change': 0, 'transition_time': '300ms'})
    multiplier = {'high': 1.2, 'medium': 1.0, 'low': 0.8}.get(word_intensity, 1.0)
    return {'rate_change': base['rate_change'] * multiplier, 'pitch_change': base['pitch_change'] * multiplier, 'transition_time': base['transition_time']}

def analyze_sentence_emotions(text):
    sentences = [s.strip() for s in re.split('[.!?]', text) if s.strip()]
    emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
    sentence_emotions = []
    prev_emotion = 'neutral'
    for sentence in sentences:
        words = sentence.split()
        word_emotions = []
        for word in words:
            word_emotion, intensity = None, None
            for emo in emotion_keywords.keys():
                intensity = get_word_emotion_intensity(word, emo)
                if intensity:
                    word_emotion = emo
                    break
            word_emotions.append({'word': word, 'emotion': word_emotion, 'intensity': intensity})
        results = emotion_analyzer(sentence)
        max_emotion = max(results[0], key=lambda x: x['score'])
        transition = calculate_transition_parameters(prev_emotion, max_emotion['label'], 'medium')
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
    emotion_stats = {}
    for e in sentence_emotions:
        emo, score = e['emotion'], e['score']
        if emo not in emotion_stats:
            emotion_stats[emo] = {'count': 0, 'high_scores': 0, 'total_score': 0, 'max_score': 0}
        emotion_stats[emo]['count'] += 1
        emotion_stats[emo]['total_score'] += score
        emotion_stats[emo]['max_score'] = max(emotion_stats[emo]['max_score'], score)
        if score > 0.8:
            emotion_stats[emo]['high_scores'] += 1
    best_emotion, best_score = None, 0
    for emo, stats in emotion_stats.items():
        weighted = stats['high_scores'] * 3 + stats['count'] * 2 + stats['total_score'] / len(sentence_emotions)
        if weighted > best_score:
            best_score = weighted
            best_emotion = emo
    return best_emotion, emotion_stats[best_emotion]['max_score']

def get_contextual_ssml(sentence_emotions):
    ssml_parts = []
    prev_emotion = None
    emotion_streak = 0
    
    for current in sentence_emotions:
        sentence = current['text']
        words = sentence.split()
        words_ssml = []
        curr_emo = current['emotion'].lower()
        keywords = emotion_keywords.get(curr_emo, [])
        
        # Aynı duygu kontrolü
        if prev_emotion == curr_emo:
            emotion_streak += 1
        else:
            emotion_streak = 0
        
        # 2. ve sonraki cümlelerde vurguyu artır
        is_emotion_intensified = emotion_streak >= 1
        
        for i, word in enumerate(words):
            word_lower = word.lower().strip(".,!?;:")
            # Son kelime değilse normal emphasis uygula
            if i < len(words) - 1:
                if word_lower in keywords:
                    # Tüm anahtar kelimeler için moderate emphasis
                    words_ssml.append(f'<emphasis level="moderate">{word}</emphasis>')
                else:
                    words_ssml.append(word)
            # Son kelime için
            else:
                if curr_emo == 'surprise':
                    # Sadece surprise cümlelerinin son kelimesi için özel ayar
                    words_ssml.append(f'<prosody pitch="+35%" rate="125%"><emphasis level="strong">{word}</emphasis></prosody>')
                else:
                    # Diğer tüm duyguların son kelimeleri için normal okuma
                    words_ssml.append(word)

        # Soru cümlesinde pitch yükseltme
        if sentence.strip().endswith('?'):
            sentence_ssml = f'<s><prosody pitch="+8%">{" ".join(words_ssml)}</prosody></s>'
        else:
            # Duyguya göre cümle tonu ayarla
            if curr_emo in emotion_settings:
                settings = emotion_settings[curr_emo]
                sentence_ssml = f'<s><prosody pitch="{settings["pitch"]}" rate="{settings["rate"]}" volume="{settings["volume"]}">{" ".join(words_ssml)}</prosody></s>'
            else:
                sentence_ssml = f'<s>{" ".join(words_ssml)}</s>'
        
        # Uzun cümlelerde otomatik nefes arası
        if len(words) > 15:
            sentence_ssml = sentence_ssml.replace(' ', ' <break time="300ms"/> ', 1)
        
        ssml_parts.append(sentence_ssml)
        prev_emotion = curr_emo
    
    # Noktalama işaretlerine göre break ekle
    ssml_body = smart_handle_punctuation(' '.join(ssml_parts))
    return f'<speak>{ssml_body}</speak>'

def smart_handle_punctuation(text):
    # Noktalama işaretlerine göre farklı break süreleri
    text = re.sub(r'\.(\s|$)', r'<break time="500ms"/>.', text)
    text = re.sub(r'!(\s|$)', r'<break time="400ms"/>!', text)
    text = re.sub(r'\?(\s|$)', r'<break time="600ms"/>?', text)
    text = re.sub(r',(\s|$)', r'<break time="200ms"/>,', text)
    text = re.sub(r';(\s|$)', r'<break time="300ms"/>;', text)
    text = re.sub(r':(\s|$)', r'<break time="250ms"/>:', text)
    text = re.sub(r'\.{3}(\s|$)', r'<break time="700ms"/>...', text)
    return text.strip()

def text_to_speech_and_emotion(text, output_file="output.mp3"):
    try:
        text = text.strip().replace('&', 'and')
        text = ''.join(char for char in text if char.isalnum() or char in ' .,!?-\'')
        sentence_emotions = analyze_sentence_emotions(text)
        overall_emotion, overall_score = determine_overall_emotion(sentence_emotions)
        speech_text = get_contextual_ssml(sentence_emotions)
        polly_client = boto3.Session(
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
            region_name='eu-central-1'
        ).client('polly')
        response = polly_client.synthesize_speech(
            Text=speech_text,
            TextType='ssml',
            OutputFormat='mp3',
            VoiceId='Joanna'
        )
        if 'AudioStream' in response:
            with open(output_file, 'wb') as file:
                file.write(response['AudioStream'].read())
        return sentence_emotions
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None