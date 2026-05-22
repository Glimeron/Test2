import streamlit as st
import pandas as pd
import joblib
import re
import yaml
import time
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

with open('config.yaml', encoding='utf-8') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# 1. ІНІЦІАЛІЗАЦІЯ ПАМ'ЯТІ
if 'bug_history' not in st.session_state:
    st.session_state.bug_history = []

# 2. Функція очищення тексту
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.lower()

# 3. Завантаження моделі (Pipeline)
@st.cache_resource
def load_model():
    return joblib.load('bug_classifier_pipeline.pkl')

try:
    pipeline = load_model()
except Exception as e:
    st.error(f"Помилка завантаження моделі: {e}. Перевірте наявність файлу 'bug_classifier_pipeline.pkl'.")
    st.stop()

# 4. НАВІГАЦІЯ ТА ІНТЕРФЕЙС
st.set_page_config(page_title="Класифікатор багів", page_icon="🛠", layout="wide")

try:
    authenticator.login(fields={
        'Form name': 'Вхід в систему класифікації дефектів',
        'Username': 'Логін',
        'Password': 'Пароль',
        'Login': 'Увійти'
    })
except Exception as e:
    st.error(e)

if st.session_state.get('authentication_status'):
    st.sidebar.write(f"Вітаємо, {st.session_state['name']}!")
    authenticator.logout('Вийти з акаунта', 'sidebar')

    st.sidebar.title("Навігація")
    page = st.sidebar.radio("Оберіть розділ:", ["📝 Нове звернення", "📊 Таблиця категорій (Історія)"])

    if page == "📝 Нове звернення":
        st.title("🛠 Інтелектуальна система класифікації звернень")
        st.markdown("""
        Цей модуль використовує алгоритм **Random Forest** та методи **NLP** для автоматичного визначення категорії баг-репорту.
        """)

        tab1, tab2 = st.tabs(["✍️ Одне звернення", "📁 Пакетна обробка (CSV)"])

        with tab1:
            user_input = st.text_area(
                "Опис проблеми (англійською):",
                height=150,
                placeholder="Введіть опис проблеми, наприклад: API request failed with timeout in Docker container..."
            )

            # ОБРОБКА КНОПКИ ТА EXPLAINABLE AI
            if st.button("Класифікувати звернення"):
                if user_input.strip():
                    with st.spinner('🤖 Аналізую текст...'):
                        
                        cleaned_input = clean_text(user_input)
                        
                        start_time = time.time()
                        
                        prediction = pipeline.predict([cleaned_input])[0]
                        probabilities = pipeline.predict_proba([cleaned_input])[0]
                        max_prob = max(probabilities) * 100
                        
                        execution_time = time.time() - start_time
                        st.info(f"⚡ Ресурси отримано з кешу. Час обробки та класифікації: {execution_time:.4f} сек.")

                        # Зберігаємо в історію
                        st.session_state.bug_history.append({
                            "Оригінальний текст": user_input,
                            "Передбачена категорія": prediction,
                            "Впевненість моделі (%)": round(max_prob, 1)
                        })

                        st.success(f"**Категорія:** {prediction}")

                        if max_prob > 80:
                            st.info(f"**Впевненість моделі:** {max_prob:.1f}%")
                        else:
                            st.warning(f"**Низька впевненість:** {max_prob:.1f}% (Потребує перевірки модератором)")

                     
                        # БЛОК: EXPLAINABLE AI (Думки ШІ)
                        with st.expander("🧠 Як ШІ дійшов цього висновку? (Логіка моделі)"):
                            st.write("**Крок 1. Відкидання словесного шуму:**")
                            st.info(f"*{cleaned_input}*")
                            
                            # Витягуємо векторизатор з нашого пайплайну
                            vectorizer = pipeline.named_steps['tfidf']
                            feature_names = vectorizer.get_feature_names_out()
                            tfidf_matrix = vectorizer.transform([cleaned_input])
                            
                            # Шукаємо слова, які модель "впізнала"
                            found_words = []
                            for col in tfidf_matrix.nonzero()[1]:
                                found_words.append(feature_names[col])
                                
                            st.write("**Крок 2. Виділення математичних маркерів (TF-IDF):**")
                            if found_words:
                                st.success(f"Модель розпізнала знайомі технічні патерни: **{', '.join(found_words)}**")
                                st.write("**Крок 3. Прийняття рішення:**")
                                st.write(f"На основі знайдених маркерів та аналізу 150 дерев рішень (Random Forest), алгоритм зіставив ці терміни з базою знань і з ймовірністю {max_prob:.1f}% відніс проблему до класу **{prediction}**.")
                            else:
                                st.warning("Модель не знайшла чітких технічних термінів. Рішення прийнято на основі непрямих ознак.")
                else:
                    st.warning("Будь ласка, введіть текст звернення.")

        with tab2:
            st.write("Завантажте CSV файл з колонкою **description** для пакетної класифікації.")
            uploaded_file = st.file_uploader("Виберіть CSV файл", type=["csv"])
            
            if uploaded_file is not None:
                try:
                    df_upload = pd.read_csv(uploaded_file)
                    if 'description' not in df_upload.columns:
                        st.error("Файл не містить обов'язкової колонки 'description'.")
                    else:
                        with st.spinner("🔄 Обробка файлу..."):
                            df_upload['cleaned'] = df_upload['description'].apply(clean_text)
                            predictions = pipeline.predict(df_upload['cleaned'])
                            probabilities = pipeline.predict_proba(df_upload['cleaned'])
                            max_probs = [max(prob) * 100 for prob in probabilities]
                            
                            df_results = pd.DataFrame({
                                "Оригінальний текст": df_upload['description'],
                                "Передбачена категорія": predictions,
                                "Впевненість моделі (%)": [round(p, 1) for p in max_probs]
                            })
                            
                            # Збереження в історію
                            for _, row in df_results.iterrows():
                                st.session_state.bug_history.append(row.to_dict())
                                
                            st.success(f"Успішно оброблено {len(df_results)} записів!")
                            st.dataframe(df_results, use_container_width=True, hide_index=True)
                except Exception as e:
                    st.error(f"Помилка читання файлу: {e}")

    elif page == "📊 Таблиця категорій (Історія)":
        st.title("📊 Історія оброблених звернень")

        if len(st.session_state.bug_history) > 0:
            df_history = pd.DataFrame(st.session_state.bug_history)
            
            # --- АНАЛІТИЧНИЙ ДАШБОРД ---
            st.subheader("📈 Аналітика")
            col1, col2, col3 = st.columns(3)
            
            total_processed = len(df_history)
            avg_confidence = df_history['Впевненість моделі (%)'].mean()
            needs_attention = len(df_history[df_history['Впевненість моделі (%)'] < 80])
            
            with col1:
                st.metric("Всього оброблено", total_processed)
            with col2:
                st.metric("Середня впевненість ШІ", f"{avg_confidence:.1f}%")
            with col3:
                st.metric("Потребують уваги (<80%)", needs_attention, delta=needs_attention if needs_attention > 0 else None, delta_color="inverse")
                
            st.bar_chart(df_history['Передбачена категорія'].value_counts())
            st.divider()

            # --- ТАБЛИЦЯ ІСТОРІЇ ---
            st.subheader("📋 Детальна історія")
            categories = ["Всі"] + list(df_history["Передбачена категорія"].unique())
            selected_category = st.selectbox("Фільтр за категорією:", categories)

            if selected_category != "Всі":
                display_history = df_history[df_history["Передбачена категорія"] == selected_category]
            else:
                display_history = df_history

            st.dataframe(display_history, use_container_width=True, hide_index=True)

            csv = display_history.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Завантажити звіт (CSV)",
                data=csv,
                file_name='bug_reports_history.csv',
                mime='text/csv',
            )
        else:
            st.info("Історія порожня. Перейдіть у розділ «Нове звернення» та класифікуйте кілька репортів.")
elif st.session_state.get('authentication_status') is False:
    st.error("Неправильний логін або пароль. Спробуйте ще раз.")
elif st.session_state.get('authentication_status') is None:
    st.warning("Будь ласка, введіть логін та пароль для доступу.")