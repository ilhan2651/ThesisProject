from transformers import pipeline
import boto3
import os
import time
import re

emotion_settings = {
    'neutral': {'rate': '100%', 'pitch': '0%', 'volume': 'medium', 'emphasis': 'none'},
    'joy': {'rate': '115%', 'pitch': '+15%', 'volume': 'x-loud', 'emphasis': 'strong'},
    'sadness': {'rate': '85%', 'pitch': '-15%', 'volume': 'soft', 'emphasis': 'reduced'},
    'fear': {'rate': '108%', 'pitch': '-5%', 'volume': 'medium', 'emphasis': 'moderate'},
    'surprise': {'rate': '108%', 'pitch': '+15%', 'volume': 'loud', 'emphasis': 'strong'},
    'disgust': {'rate': '97%', 'pitch': '-10%', 'volume': 'medium', 'emphasis': 'strong'},
    'anger': {'rate': '110%', 'pitch': '+8%', 'volume': 'loud', 'emphasis': 'strong'}
}

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

emotional_words = {
    'fear': ['terrified', 'scared', 'afraid', 'frightened', 'horrified', 'panicked', 'trembling', 'shaking', 'terror', 'horror',
             'dread', 'panic', 'threat', 'danger', 'scary', 'dark', 'shadow', 'nightmare', 'scream', 'hide', 'run', 'escape'],
    'anger': ['angry', 'furious', 'rage', 'outraged', 'mad', 'hate', 'destroy', 'revenge', 'fight', 'enough', 'stop', 'never',
              'dare', 'wrong', 'fault', 'blame', 'unfair', 'betrayed', 'frustrated', 'irritated', 'annoyed', 'hostile']
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
    adjectives = ['good', 'great', 'happy', 'beautiful', 'amazing', 'wonderful', 'lovely', 'nice', 'excellent', 'fantastic']
    words = text.split()
    modified_words = []
    for word in words:
        word_lower = word.lower()
        if word_lower in adjectives:
            vowels = 'aeiou'
            mid_point = len(word) // 2
            for i in range(mid_point-1, -1, -1):
                if word[i].lower() in vowels:
                    word = word[:i] + word[i]*3 + word[i+1:]
                    break
        modified_words.append(word)
    return ' '.join(modified_words)

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
    if emotion_type in emotion_words:
        if word in emotion_words[emotion_type]['high_intensity']: return 'high'
        elif word in emotion_words[emotion_type]['medium_intensity']: return 'medium'
        elif word in emotion_words[emotion_type]['low_intensity']: return 'low'
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
            for emo in emotion_words.keys():
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

def get_contextual_ssml(sentence_emotions, overall_emotion):
    ssml_parts = []
    for current in sentence_emotions:
        words = current['text'].split()
        words_ssml = []
        curr_emo = current['emotion'].lower()
        for word in words:
            if curr_emo == 'fear' and word.lower() in emotional_words['fear']:
                words_ssml.append(f"""<prosody pitch="-10%" rate="105%"><emphasis level="moderate">{word}</emphasis></prosody>""")
            elif curr_emo == 'anger' and word.lower() in emotional_words['anger']:
                words_ssml.append(f"""<prosody pitch="+10%" volume="loud"><emphasis level="strong">{word}</emphasis></prosody>""")
            else:
                words_ssml.append(word)
        settings = emotion_settings.get(curr_emo, emotion_settings['neutral'])
        sentence_ssml = f"""<prosody rate="{settings['rate']}" pitch="{settings['pitch']}" volume="{settings['volume']}">
            {' '.join(words_ssml)}</prosody><break time="200ms"/>"""
        ssml_parts.append(sentence_ssml)
    return f"""<speak>{' '.join(ssml_parts)}</speak>"""

def text_to_speech_and_emotion(text, output_file="output.mp3"):
    try:
        text = text.strip().replace('&', 'and')
        text = ''.join(char for char in text if char.isalnum() or char in ' .,!?-\'')
        sentence_emotions = analyze_sentence_emotions(text)
        overall_emotion, overall_score = determine_overall_emotion(sentence_emotions)
        speech_text = get_contextual_ssml(sentence_emotions, overall_emotion)
        polly_client = boto3.Session(
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
            region_name='eu-central-1'
        ).client('polly')
        response = polly_client.synthesize_speech(
            Text=speech_text,
            TextType='ssml',
            OutputFormat='mp3',
            VoiceId='Kimberly'
        )
        if 'AudioStream' in response:
            with open(output_file, 'wb') as file:
                file.write(response['AudioStream'].read())
        return sentence_emotions
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None
