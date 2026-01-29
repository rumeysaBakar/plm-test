import streamlit as st
import json
from openai import OpenAI
import PyPDF2
import io
import os
from dotenv import load_dotenv
load_dotenv()
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
        model="gpt-5.1",
        messages=[
            {"role": "system", "content": extraction_prompt},
            {"role": "user", "content": all_content}
        ],
        temperature=0.3,
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)


def generate_raw_response(question: str, plm_profile: dict, client: OpenAI) -> str:

    knowledge = plm_profile.get("knowledge_layer", {})
    reasoning = plm_profile.get("reasoning_layer", {})

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

Soruyu bu profile uygun şekilde, içerik olarak doğru ve kapsamlı yanıtla.
Muhakeme profiline göre kesin veya belirsiz ifadeler kullan.
"""

    response = client.chat.completions.create(
        model="gpt-5.1",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.7
    )

    return response.choices[0].message.content


def apply_plm_rewrite(raw_response: str, plm_profile: dict, client: OpenAI) -> str:

    language = plm_profile.get("language_layer", {})
    persona = plm_profile.get("persona_summary", "")

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

PERSONA: {persona}

KURALLAR:
1. İçeriğin anlamını koru
2. Cümle yapısını profile göre ayarla
3. Karakteristik kalıpları doğal şekilde ekle
4. Tonu tutarlı tut
5. Gerçek bir insan yazmış gibi görünmeli

HAM CEVAP:
{raw_response}

YENİDEN YAZILMIŞ CEVAP:
"""

    response = client.chat.completions.create(
        model="gpt-5.1",
        messages=[
            {"role": "user", "content": rewrite_prompt}
        ],
        temperature=0.8
    )

    return response.choices[0].message.content


def generate_twin_response(question: str, plm_profile: dict, client: OpenAI) -> tuple[str, str]:


    raw_response = generate_raw_response(question, plm_profile, client)

    plm_response = apply_plm_rewrite(raw_response, plm_profile, client)

    return raw_response, plm_response



def render_step_indicator():
    """Adım göstergesi"""
    cols = st.columns(3)
    steps = [" Profil Bilgileri", " Döküman Yükleme", " AI Twin"]

    for i, (col, step) in enumerate(zip(cols, steps), 1):
        with col:
            if i == st.session_state.step:
                st.markdown(f"** {step}**")
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

    # Profil özeti
    with st.expander(" Profil Özetiniz", expanded=False):
        st.json(st.session_state.profile_data)

    # Dosya yükleme
    uploaded_files = st.file_uploader(
        "PDF, TXT veya metin dosyaları yükleyin",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Şu an demo için sadece metin tabanlı dosyalar destekleniyor."
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} dosya seçildi")

        # Dosya içeriklerini işle
        content_list = []
        for file in uploaded_files:
            st.write(f" {file.name}")

            if file.type == "application/pdf":
                text = extract_pdf_text(file)
                content_list.append(f"[{file.name}]:\n{text[:2000]}...")  # İlk 2000 karakter
            else:
                text = file.read().decode("utf-8", errors="ignore")
                content_list.append(f"[{file.name}]:\n{text[:2000]}...")

        st.session_state.uploaded_content = content_list

    st.divider()
    st.subheader("  Manuel Metin ")
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
        with st.spinner("PLM Profili oluşturuluyor..."):
            try:
                st.session_state.plm_profile = build_plm_profile(
                    st.session_state.profile_data,
                    st.session_state.uploaded_content,
                    client
                )
                st.success(" PLM Profili oluşturuldu!")
            except Exception as e:
                st.error(f"Hata: {e}")
                return

    # PLM Profil görüntüleme
    with st.expander(" PLM Profil Detayları", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader(" Bilgi Katmanı")
            st.json(st.session_state.plm_profile.get("knowledge_layer", {}))

        with col2:
            st.subheader(" Muhakeme Katmanı")
            st.json(st.session_state.plm_profile.get("reasoning_layer", {}))

        with col3:
            st.subheader(" Dil/Ton Katmanı")
            st.json(st.session_state.plm_profile.get("language_layer", {}))

        st.info(f"**Persona:** {st.session_state.plm_profile.get('persona_summary', '')}")

    st.divider()

    st.subheader(f" {st.session_state.profile_data.get('name', 'AI Twin')} ile Konuşun")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and "raw" in msg:
                tab1, tab2 = st.tabs([" PLM Çıktısı", " Ham Çıktı"])
                with tab1:
                    st.write(msg["content"])
                with tab2:
                    st.write(msg["raw"])
            else:
                st.write(msg["content"])

    if prompt := st.chat_input("AI Twin'inize bir soru sorun..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Düşünüyor..."):
                try:
                    raw_response, plm_response = generate_twin_response(
                        prompt,
                        st.session_state.plm_profile,
                        client
                    )

                    tab1, tab2 = st.tabs([" PLM Çıktısı", " Ham Çıktı"])
                    with tab1:
                        st.write(plm_response)
                    with tab2:
                        st.write(raw_response)
                        st.caption(
                            "️ Bu, PLM olmadan üretilen ham cevap. PLM katmanı bunu kişinin diline göre yeniden yazıyor.")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": plm_response,
                        "raw": raw_response
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
