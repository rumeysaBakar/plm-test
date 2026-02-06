import streamlit as st
import json
from openai import OpenAI
import PyPDF2
import io
import os
import difflib
import re
from collections import Counter
from dotenv import load_dotenv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

st.set_page_config(
    page_title="AI Twin Demo",
    page_icon="",
    layout="wide"
)

if "step" not in st.session_state:
    st.session_state.step = 1
if "profile_data" not in st.session_state:
    st.session_state.profile_data = {}
if "uploaded_content" not in st.session_state:
    st.session_state.uploaded_content = []
if "plm_profile" not in st.session_state:
    st.session_state.plm_profile = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = []



def analyze_text_metrics(text: str) -> dict:
    """Metin metriklerini hesaplama"""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    words = text.split()

    # Ortalama cümle uzunluğu
    avg_sentence_length = len(words) / len(sentences) if sentences else 0

    # Kelime çeşitliliği
    word_diversity = len(set(words)) / len(words) if words else 0

    formal_words = ['ancak', 'bununla birlikte', 'dolayısıyla', 'nitekim', 'öte yandan',
                    'sonuç olarak', 'bu bağlamda', 'şöyle ki', 'binaenaleyh']
    casual_words = ['yani', 'aslında', 'hani', 'işte', 'mesela', 'falan', 'şey', 'ama']

    text_lower = text.lower()
    formal_count = sum(1 for w in formal_words if w in text_lower)
    casual_count = sum(1 for w in casual_words if w in text_lower)

    hedging_words = ['belki', 'muhtemelen', 'sanırım', 'galiba', 'olabilir', 'düşünüyorum',
                     'tahminimce', 'gibi görünüyor', 'bir bakıma']
    definitive_words = ['kesinlikle', 'mutlaka', 'şüphesiz', 'elbette', 'tabii ki',
                        'kuşkusuz', 'açıkça', 'net olarak']

    hedging_count = sum(1 for w in hedging_words if w in text_lower)
    definitive_count = sum(1 for w in definitive_words if w in text_lower)

    emoji_count = len(re.findall(r'[^\w\s,.\-:;\'\"()\[\]{}!?]', text))
    exclamation_count = text.count('!')

    return {
        'total_words': len(words),
        'total_sentences': len(sentences),
        'avg_sentence_length': round(avg_sentence_length, 1),
        'word_diversity': round(word_diversity * 100, 1),
        'formal_markers': formal_count,
        'casual_markers': casual_count,
        'hedging_markers': hedging_count,
        'definitive_markers': definitive_count,
        'emoji_count': emoji_count,
        'exclamation_count': exclamation_count
    }


def get_word_diff(text1: str, text2: str) -> list:
    """İki metin arasındaki kelime bazlı farklar"""
    words1 = text1.split()
    words2 = text2.split()

    matcher = difflib.SequenceMatcher(None, words1, words2)

    diff_result = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            diff_result.append({
                'type': 'replace',
                'old': ' '.join(words1[i1:i2]),
                'new': ' '.join(words2[j1:j2])
            })
        elif tag == 'delete':
            diff_result.append({
                'type': 'delete',
                'old': ' '.join(words1[i1:i2]),
                'new': ''
            })
        elif tag == 'insert':
            diff_result.append({
                'type': 'insert',
                'old': '',
                'new': ' '.join(words2[j1:j2])
            })

    return diff_result


def analyze_plm_changes(raw: str, plm: str, client: OpenAI) -> dict:

    analysis_prompt = f"""
İki metin arasındaki stil farklarını analiz et:

HAM METİN:
{raw}

PLM METİN:
{plm}

Aşağıdaki kategorilerde değişiklikleri tespit et ve JSON formatında yanıtla:

{{
    "tone_changes": [
        {{"original": "orijinal ifade", "changed_to": "değiştirilmiş ifade", "reason": "neden değiştirildi"}}
    ],
    "formality_changes": [
        {{"original": "orijinal", "changed_to": "yeni", "direction": "more_formal/less_formal"}}
    ],
    "sentence_structure_changes": [
        {{"description": "ne değişti", "example": "örnek"}}
    ],
    "added_phrases": ["eklenen karakteristik ifadeler"],
    "removed_phrases": ["çıkarılan ifadeler"],
    "certainty_changes": [
        {{"original": "orijinal", "changed_to": "yeni", "direction": "more_certain/less_certain"}}
    ],
    "overall_summary": "Genel olarak PLM metni nasıl farklılaştırdı - 2-3 cümle"
}}
"""

    response = client.chat.completions.create(
        model="gpt-5.1-chat-latest",
        messages=[{"role": "user", "content": analysis_prompt}],
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)


def render_diff_analysis(raw: str, plm: str, ai_analysis: dict = None):
    """Diff analizini görselleştir"""

    st.markdown("###  PLM Dönüşüm Analizi")

    # Metrik karşılaştırması
    raw_metrics = analyze_text_metrics(raw)
    plm_metrics = analyze_text_metrics(plm)

    # Metrik kartları
    st.markdown("####  Sayısal Karşılaştırma")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        delta = plm_metrics['avg_sentence_length'] - raw_metrics['avg_sentence_length']
        st.metric(
            "Ort. Cümle Uzunluğu",
            f"{plm_metrics['avg_sentence_length']} kelime",
            delta=f"{delta:+.1f}",
            delta_color="off"
        )

    with col2:
        delta = plm_metrics['formal_markers'] - raw_metrics['formal_markers']
        st.metric(
            "Formal İfadeler",
            plm_metrics['formal_markers'],
            delta=f"{delta:+d}",
            delta_color="off"
        )

    with col3:
        delta = plm_metrics['casual_markers'] - raw_metrics['casual_markers']
        st.metric(
            "Günlük İfadeler",
            plm_metrics['casual_markers'],
            delta=f"{delta:+d}",
            delta_color="off"
        )

    with col4:
        delta = plm_metrics['hedging_markers'] - raw_metrics['hedging_markers']
        st.metric(
            "Belirsizlik İfadeleri",
            plm_metrics['hedging_markers'],
            delta=f"{delta:+d}",
            delta_color="off"
        )

    # Detaylı metrik tablosu
    with st.expander(" Tüm Metrikler", expanded=False):
        metrics_data = {
            "Metrik": [
                "Toplam Kelime",
                "Toplam Cümle",
                "Ort. Cümle Uzunluğu",
                "Kelime Çeşitliliği (%)",
                "Formal İfadeler",
                "Günlük İfadeler",
                "Belirsizlik İfadeleri",
                "Kesinlik İfadeleri",
                "Ünlem Sayısı"
            ],
            "Ham Çıktı": [
                raw_metrics['total_words'],
                raw_metrics['total_sentences'],
                raw_metrics['avg_sentence_length'],
                raw_metrics['word_diversity'],
                raw_metrics['formal_markers'],
                raw_metrics['casual_markers'],
                raw_metrics['hedging_markers'],
                raw_metrics['definitive_markers'],
                raw_metrics['exclamation_count']
            ],
            "PLM Çıktı": [
                plm_metrics['total_words'],
                plm_metrics['total_sentences'],
                plm_metrics['avg_sentence_length'],
                plm_metrics['word_diversity'],
                plm_metrics['formal_markers'],
                plm_metrics['casual_markers'],
                plm_metrics['hedging_markers'],
                plm_metrics['definitive_markers'],
                plm_metrics['exclamation_count']
            ],
            "Fark": [
                plm_metrics['total_words'] - raw_metrics['total_words'],
                plm_metrics['total_sentences'] - raw_metrics['total_sentences'],
                round(plm_metrics['avg_sentence_length'] - raw_metrics['avg_sentence_length'], 1),
                round(plm_metrics['word_diversity'] - raw_metrics['word_diversity'], 1),
                plm_metrics['formal_markers'] - raw_metrics['formal_markers'],
                plm_metrics['casual_markers'] - raw_metrics['casual_markers'],
                plm_metrics['hedging_markers'] - raw_metrics['hedging_markers'],
                plm_metrics['definitive_markers'] - raw_metrics['definitive_markers'],
                plm_metrics['exclamation_count'] - raw_metrics['exclamation_count']
            ]
        }
        st.dataframe(metrics_data, use_container_width=True)

    # Kelime bazlı diff
    st.markdown("####  Kelime Bazlı Değişiklikler")

    word_diffs = get_word_diff(raw, plm)

    if word_diffs:
        changes_found = False
        for diff in word_diffs[:10]:
            if diff['type'] == 'replace':
                changes_found = True
                col1, col2, col3 = st.columns([2, 1, 2])
                with col1:
                    st.markdown(f"~~{diff['old'][:50]}...~~" if len(diff['old']) > 50 else f"~~{diff['old']}~~")
                with col2:
                    st.markdown("➡️")
                with col3:
                    st.markdown(f"**{diff['new'][:50]}...**" if len(diff['new']) > 50 else f"**{diff['new']}**")

        if not changes_found:
            st.info("Kelime bazlı büyük değişiklik tespit edilmedi.")

    # AI Analizi
    if ai_analysis:
        st.markdown("####  AI Değişiklik Analizi")

        # Genel özet
        st.info(f"**Özet:** {ai_analysis.get('overall_summary', 'Analiz yapılamadı')}")

        # Ton değişiklikleri
        tone_changes = ai_analysis.get('tone_changes', [])
        if tone_changes:
            with st.expander(f" Ton Değişiklikleri ({len(tone_changes)})", expanded=True):
                for change in tone_changes[:5]:
                    st.markdown(f"""
                    - **Orijinal:** {change.get('original', '')}
                    - **Yeni:** {change.get('changed_to', '')}
                    - **Sebep:** _{change.get('reason', '')}_
                    """)
                    st.divider()

        # Formalite değişiklikleri
        formality_changes = ai_analysis.get('formality_changes', [])
        if formality_changes:
            with st.expander(f" Formalite Değişiklikleri ({len(formality_changes)})"):
                for change in formality_changes[:5]:
                    direction = " Daha Formal" if change.get('direction') == 'more_formal' else " Daha Günlük"
                    st.markdown(f"- {direction}: _{change.get('original', '')}_ → **{change.get('changed_to', '')}**")

        # Kesinlik değişiklikleri
        certainty_changes = ai_analysis.get('certainty_changes', [])
        if certainty_changes:
            with st.expander(f" Kesinlik Değişiklikleri ({len(certainty_changes)})"):
                for change in certainty_changes[:5]:
                    direction = "Daha Kesin" if change.get('direction') == 'more_certain' else "️ Daha Belirsiz"
                    st.markdown(f"- {direction}: _{change.get('original', '')}_ → **{change.get('changed_to', '')}**")

        # Eklenen ifadeler
        added = ai_analysis.get('added_phrases', [])
        if added:
            with st.expander(f"➕ Eklenen İfadeler ({len(added)})"):
                for phrase in added:
                    st.markdown(f"- **{phrase}**")

        # Çıkarılan ifadeler
        removed = ai_analysis.get('removed_phrases', [])
        if removed:
            with st.expander(f"➖ Çıkarılan İfadeler ({len(removed)})"):
                for phrase in removed:
                    st.markdown(f"- ~~{phrase}~~")



def extract_style_examples(uploaded_content: list, client: OpenAI) -> list:

    if not uploaded_content:
        return []

    sample_text = " ".join(uploaded_content)[:3000]

    extraction_prompt = """
Aşağıdaki metinden, yazarın karakteristik 3-5 cümle örneği çıkar.
Bu cümleler yazarın tipik cümle yapısını, kullandığı bağlaçları ve ifade tarzını yansıtmalı.

Metin:
""" + sample_text + """

Sadece cümleleri liste halinde ver, başka açıklama yapma:
1. [cümle]
2. [cümle]
3. [cümle]
"""

    try:
        response = client.chat.completions.create(
            model="gpt-5.1-chat-latest",
            messages=[{"role": "user", "content": extraction_prompt}]

        )

        content = response.choices[0].message.content
        examples = []
        for line in content.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                cleaned = re.sub(r'^[\d\-\.\)\s]+', '', line).strip()
                if cleaned and len(cleaned) > 10:
                    examples.append(cleaned)

        return examples[:5]  # Max 5 örnek
    except:
        return []


def calculate_similarity(text1: str, text2: str) -> float:
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return float(similarity[0][0])
    except:
        return 0.0


def is_in_expertise_area(question: str, expertise_areas: list) -> bool:
    if not expertise_areas:
        return False

    question_lower = question.lower()
    expertise_text = " ".join(expertise_areas).lower()

    question_words = set(question_lower.split())
    expertise_words = set(expertise_text.split())

    common_words = question_words.intersection(expertise_words)
    if len(common_words) > 0:
        return True

    # Veya kosinüs benzerliği
    similarity = calculate_similarity(question_lower, expertise_text)
    return similarity > 0.1


def get_conversation_context() -> str:
    recent_messages = st.session_state.conversation_memory[-3:]
    if not recent_messages:
        return ""

    context_parts = []
    for msg in recent_messages:
        role = "Kullanıcı" if msg["role"] == "user" else "Asistan"
        context_parts.append(f"{role}: {msg['content'][:200]}...")

    return "\n".join(context_parts)



def extract_pdf_text(pdf_file) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"[PDF okuma hatası: {e}]"


def build_plm_profile(profile_data: dict, uploaded_content: list, client: OpenAI) -> dict:
    all_content = f"""
KULLANICI PROFİL BİLGİLERİ:
- Ad Soyad: {profile_data.get('name', 'Belirtilmedi')}
- Yaş: {profile_data.get('age', 'Belirtilmedi')}
- Meslek: {profile_data.get('profession', 'Belirtilmedi')}
- Deneyim: {profile_data.get('experience_years', 'Belirtilmedi')} yıl
- Yaptığı Çalışmalar: {profile_data.get('works', 'Belirtilmedi')}
- En Büyük Başarısı: {profile_data.get('achievement', 'Belirtilmedi')}
- Uzmanlık Alanları: {profile_data.get('expertise', 'Belirtilmedi')}
- Neden Uzman: {profile_data.get('why_expert', 'Belirtilmedi')}

YÜKLENEN DÖKÜMANLARDAN İÇERİK:
{chr(10).join(uploaded_content) if uploaded_content else 'Döküman yüklenmedi'}
"""

    extraction_prompt = """
Sen bir PLM (Personal Language Model) profil analisti olarak çalışıyorsun. 
Verilen kullanıcı bilgilerinden 3 katmanlı bir kişilik profili çıkarmalısın.

ÇIKARMANI GEREKEN 3 KATMAN:

1. **KNOWLEDGE_LAYER (Bilgi Katmanı)**
   - Kişinin uzmanlık alanları
   - Bildiği konular ve derinlik seviyeleri
   - Sektörel bilgisi
   - Teknik yetkinlikleri

2. **REASONING_LAYER (Muhakeme Katmanı)**
   - Problem çözme yaklaşımı
   - Karar verme tarzı (temkinli mi, hızlı mı?)
   - Hangi konularda kesin konuşur, hangilerinde belirsiz kalır?
   - Analitik mi yoksa sezgisel mi?
   - Risk toleransı

3. **LANGUAGE_LAYER (Dil/Ton Katmanı)**
   - Cümle uzunluğu tercihi (kısa/orta/uzun)
   - Formalite seviyesi (resmi/yarı-resmi/samimi)
   - Kesinlik derecesi (kesin ifadeler mi, yumuşatılmış mı?)
   - Karakteristik kelimeler veya kalıplar
   - Açıklama tarzı (örneklerle mi, teorik mi, pratik mi?)
   - Emoji/ünlem kullanımı

Aşağıdaki JSON formatında yanıt ver (başka hiçbir şey yazma):

{
    "knowledge_layer": {
        "primary_expertise": ["alan1", "alan2"],
        "secondary_knowledge": ["alan1", "alan2"],
        "depth_level": "beginner/intermediate/expert/master",
        "industry_context": "sektör bilgisi"
    },
    "reasoning_layer": {
        "decision_style": "analytical/intuitive/balanced",
        "confidence_areas": ["kesin konuştuğu alanlar"],
        "uncertain_areas": ["belirsiz kaldığı alanlar"],
        "problem_approach": "systematic/creative/pragmatic",
        "risk_tolerance": "low/medium/high"
    },
    "language_layer": {
        "sentence_length": "short/medium/long",
        "formality": "formal/semi-formal/casual",
        "certainty_level": "definitive/hedged/mixed",
        "explanation_style": "examples/theoretical/practical",
        "characteristic_phrases": ["örnek kalıp1", "örnek kalıp2"],
        "tone": "professional/friendly/authoritative/humble"
    },
    "persona_summary": "Bu kişinin tek cümlelik özeti"
}
"""

    response = client.chat.completions.create(
        model="gpt-5.1-chat-latest",
        messages=[
            {"role": "system", "content": extraction_prompt},
            {"role": "user", "content": all_content}
        ],

        response_format={"type": "json_object"}
    )

    profile = json.loads(response.choices[0].message.content)

    style_examples = extract_style_examples(uploaded_content, client)
    if style_examples:
        profile["language_layer"]["style_examples"] = style_examples

    return profile


def generate_raw_response(question: str, plm_profile: dict, client: OpenAI) -> str:
    knowledge = plm_profile.get("knowledge_layer", {})
    reasoning = plm_profile.get("reasoning_layer", {})

    expertise_areas = knowledge.get('primary_expertise', []) + knowledge.get('secondary_knowledge', [])
    in_expertise = is_in_expertise_area(question, expertise_areas)

    dynamic_certainty = "definitive" if in_expertise else "hedged"

    conversation_context = get_conversation_context()
    memory_context = ""
    if conversation_context:
        memory_context = f"\nÖNCEKİ KONUŞMA BAĞLAMI:\n{conversation_context}\n"

    system_prompt = f"""
Sen bir uzman asistanısın. Aşağıdaki bilgi ve muhakeme profiline göre soruyu yanıtla.

BİLGİ PROFİLİ:
- Ana Uzmanlık: {knowledge.get('primary_expertise', [])}
- İkincil Bilgi: {knowledge.get('secondary_knowledge', [])}
- Derinlik: {knowledge.get('depth_level', 'intermediate')}
- Sektör: {knowledge.get('industry_context', '')}

MUHAKEME PROFİLİ:
- Karar Tarzı: {reasoning.get('decision_style', 'balanced')}
- Kesin Olduğu Alanlar: {reasoning.get('confidence_areas', [])}
- Belirsiz Alanlar: {reasoning.get('uncertain_areas', [])}
- Problem Yaklaşımı: {reasoning.get('problem_approach', 'pragmatic')}
- Dinamik Kesinlik Seviyesi: {dynamic_certainty} (Soru uzmanlık alanında: {in_expertise})
{memory_context}

Soruyu bu profile uygun şekilde, içerik olarak doğru ve kapsamlı yanıtla.
Muhakeme profiline göre kesin veya belirsiz ifadeler kullan.
"""

    response = client.chat.completions.create(
        model="gpt-5.1-chat-latest",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]

    )

    return response.choices[0].message.content


def apply_plm_rewrite(raw_response: str, plm_profile: dict, client: OpenAI) -> str:
    language = plm_profile.get("language_layer", {})
    persona = plm_profile.get("persona_summary", "")

    style_examples = language.get('style_examples', [])
    examples_text = ""
    if style_examples:
        examples_text = "\nKULLANICININ KARAKTERİSTİK CÜMLE ÖRNEKLERİ (bu tarzı taklit et):\n" + \
                        "\n".join([f"- {ex}" for ex in style_examples[:3]])

    rewrite_prompt = f"""
Sen bir PLM (Personal Language Model) yeniden yazım motorusun.

GÖREV: Aşağıdaki ham cevabı, belirtilen dil profiline göre yeniden yaz.
İçeriği DEĞIŞTIRME, sadece NASIL söylendiğini değiştir.

DİL PROFİLİ:
- Cümle Uzunluğu: {language.get('sentence_length', 'medium')}
- Formalite: {language.get('formality', 'semi-formal')}
- Kesinlik Seviyesi: {language.get('certainty_level', 'mixed')}
- Açıklama Tarzı: {language.get('explanation_style', 'practical')}
- Karakteristik Kalıplar: {language.get('characteristic_phrases', [])}
- Ton: {language.get('tone', 'professional')}
{examples_text}

PERSONA: {persona}

KURALLAR:
1. İçeriğin anlamını koru
2. Cümle yapısını profile göre ayarla
3. Karakteristik kalıpları doğal şekilde ekle
4. Yukarıdaki cümle örneklerindeki yapıyı taklit et
5. Tonu tutarlı tut
6. Gerçek bir insan yazmış gibi görünmeli

YASAKLAR:
- "Buna göre", "Sonuç olarak" gibi generic bağlaçlar kullanma (eğer kullanıcı kullanmıyorsa)
- Profildeki formalite seviyesi dışında bir ton kullanma

HAM CEVAP:
{raw_response}

YENİDEN YAZILMIŞ CEVAP:
"""

    response = client.chat.completions.create(
        model="gpt-5.1-chat-latest",
        messages=[
            {"role": "user", "content": rewrite_prompt}
        ]
    )

    return response.choices[0].message.content


def generate_twin_response(question: str, plm_profile: dict, client: OpenAI) -> tuple[str, str, dict]:

    raw_response = generate_raw_response(question, plm_profile, client)
    plm_response = apply_plm_rewrite(raw_response, plm_profile, client)

    # AI analizi yap
    try:
        ai_analysis = analyze_plm_changes(raw_response, plm_response, client)
    except:
        ai_analysis = None

    return raw_response, plm_response, ai_analysis



def render_step_indicator():
    """Adım göstergesi"""
    cols = st.columns(3)
    steps = ["Profil Bilgileri", " Döküman Yükleme", " AI Twin"]

    for i, (col, step) in enumerate(zip(cols, steps), 1):
        with col:
            if i == st.session_state.step:
                st.markdown(f" {step}")
            elif i < st.session_state.step:
                st.markdown(f" {step}")
            else:
                st.markdown(f" {step}")

    st.divider()


def render_step1():
    """Adım 1: Profil Soruları"""
    st.header(" Kendinizi Tanıtın")
    st.caption("AI ikizinizi oluşturmak için size birkaç soru soracağız.")

    with st.form("profile_form"):
        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("Adınız Soyadınız *", value=st.session_state.profile_data.get("name", ""))
            age = st.number_input("Yaşınız", min_value=18, max_value=100,
                                  value=st.session_state.profile_data.get("age", 30))
            profession = st.text_input("Mesleğiniz *", value=st.session_state.profile_data.get("profession", ""))
            experience_years = st.number_input("Bu meslekte kaç yıldır çalışıyorsunuz?", min_value=0, max_value=60,
                                               value=st.session_state.profile_data.get("experience_years", 5))

        with col2:
            works = st.text_area("Bu zamana kadar yaptığınız önemli çalışmalar neler?",
                                 value=st.session_state.profile_data.get("works", ""), height=100)
            achievement = st.text_area("Mesleğinizdeki en büyük başarınız ne?",
                                       value=st.session_state.profile_data.get("achievement", ""), height=100)

        expertise = st.text_area("Uzman olduğunuz konular neler?",
                                 value=st.session_state.profile_data.get("expertise", ""), height=80)
        why_expert = st.text_area("Bu konularda neden uzman olduğunuzu düşünüyorsunuz?",
                                  value=st.session_state.profile_data.get("why_expert", ""), height=80)

        submitted = st.form_submit_button("İleri →", use_container_width=True, type="primary")

        if submitted:
            if not name or not profession:
                st.error("Lütfen zorunlu alanları doldurun!")
            else:
                st.session_state.profile_data = {
                    "name": name,
                    "age": age,
                    "profession": profession,
                    "experience_years": experience_years,
                    "works": works,
                    "achievement": achievement,
                    "expertise": expertise,
                    "why_expert": why_expert
                }
                st.session_state.step = 2
                st.rerun()


def render_step2():
    st.header(" Dökümanlarınızı Yükleyin")
    st.caption(
        "Mesleğinizle ve kendinizle alakalı dökümanlar yükleyin. Bu dökümanlar AI ikizinizin bilgi tabanını oluşturacak.")

    with st.expander("📋 Profil Özetiniz", expanded=False):
        st.json(st.session_state.profile_data)

    uploaded_files = st.file_uploader(
        "PDF, TXT veya metin dosyaları yükleyin",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Şu an demo için sadece metin tabanlı dosyalar destekleniyor."
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} dosya seçildi")

        content_list = []
        for file in uploaded_files:
            st.write(f" {file.name}")

            if file.type == "application/pdf":
                text = extract_pdf_text(file)
                content_list.append(f"[{file.name}]:\n{text[:2000]}...")
            else:
                text = file.read().decode("utf-8", errors="ignore")
                content_list.append(f"[{file.name}]:\n{text[:2000]}...")

        st.session_state.uploaded_content = content_list

    st.divider()
    st.subheader(" Manuel Metin")
    manual_text = st.text_area(
        "Kendinizi anlatan, yazı stilinizi gösteren örnek metinler ekleyin",
        height=150,
        placeholder="Örneğin: Daha önce yazdığınız makaleler, blog yazıları, e-postalar..."
    )

    if manual_text:
        if manual_text not in st.session_state.uploaded_content:
            st.session_state.uploaded_content.append(f"[Manuel Giriş]:\n{manual_text}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Geri", use_container_width=True):
            st.session_state.step = 1
            st.rerun()
    with col2:
        if st.button("AI Twin Oluştur →", use_container_width=True, type="primary"):
            st.session_state.step = 3
            st.rerun()


def render_step3():
    st.header(" AI Twin'iniz Hazır")

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.error(" OPENAI_API_KEY ortam değişkeni bulunamadı.")
        st.info("Lütfen .env dosyasını kontrol edin.")
        return

    client = OpenAI(api_key=api_key)

    if st.session_state.plm_profile is None:
        with st.spinner(" PLM Profili oluşturuluyor..."):
            try:
                st.session_state.plm_profile = build_plm_profile(
                    st.session_state.profile_data,
                    st.session_state.uploaded_content,
                    client
                )
                st.success(" PLM Profili oluşturuldu!")

                style_examples = st.session_state.plm_profile.get("language_layer", {}).get("style_examples", [])
                if style_examples:
                    with st.expander(" Tespit Edilen Stil Örnekleri", expanded=True):
                        for i, ex in enumerate(style_examples, 1):
                            st.markdown(f"{i}. {ex}")

            except Exception as e:
                st.error(f"Hata: {e}")
                return

    with st.expander(" PLM Profil Detayları", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Bilgi Katmanı")
            st.json(st.session_state.plm_profile.get("knowledge_layer", {}))

        with col2:
            st.subheader(" Muhakeme Katmanı")
            st.json(st.session_state.plm_profile.get("reasoning_layer", {}))

        with col3:
            st.subheader(" Dil/Ton Katmanı")
            lang_layer = st.session_state.plm_profile.get("language_layer", {}).copy()
            if "style_examples" in lang_layer:
                lang_layer["style_examples"] = f"[{len(lang_layer['style_examples'])} örnek]"
            st.json(lang_layer)

        st.info(f"**Persona:** {st.session_state.plm_profile.get('persona_summary', '')}")

    st.divider()

    st.subheader(f"{st.session_state.profile_data.get('name', 'AI Twin')} ile Konuşun")

    # Mesaj geçmişi
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and "raw" in msg:
                tab1, tab2, tab3 = st.tabs([" PLM Çıktısı", " Ham Çıktı", " Fark Analizi"])
                with tab1:
                    st.write(msg["content"])
                with tab2:
                    st.write(msg["raw"])
                with tab3:
                    render_diff_analysis(msg["raw"], msg["content"], msg.get("analysis"))
            else:
                st.write(msg["content"])

    # Yeni mesaj
    if prompt := st.chat_input("AI Twin'inize bir soru sorun..."):
        st.session_state.conversation_memory.append({"role": "user", "content": prompt})

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Düşünüyor..."):
                try:
                    raw_response, plm_response, ai_analysis = generate_twin_response(
                        prompt,
                        st.session_state.plm_profile,
                        client
                    )

                    tab1, tab2, tab3 = st.tabs([" PLM Çıktısı", " Ham Çıktı", " Fark Analizi"])
                    with tab1:
                        st.write(plm_response)
                    with tab2:
                        st.write(raw_response)
                        st.caption("️ Bu, PLM olmadan üretilen ham cevap.")
                    with tab3:
                        render_diff_analysis(raw_response, plm_response, ai_analysis)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": plm_response,
                        "raw": raw_response,
                        "analysis": ai_analysis
                    })

                    st.session_state.conversation_memory.append({
                        "role": "assistant",
                        "content": plm_response
                    })

                except Exception as e:
                    st.error(f"Hata: {e}")

    st.sidebar.divider()
    if st.sidebar.button(" Baştan Başla", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


def main():
    st.title(" AI Twin Demo")
    st.caption("Personal Language Model (PLM) Katmanı Demonstrasyonu")

    render_step_indicator()

    if st.session_state.step == 1:
        render_step1()
    elif st.session_state.step == 2:
        render_step2()
    elif st.session_state.step == 3:
        render_step3()


if __name__ == "__main__":
    main()
