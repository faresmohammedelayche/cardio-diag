@st.cache_resource
def load_model():
    model_path = "model.keras"
    if os.path.exists(model_path):
        try:
            import tensorflow as tf
            import keras
            
            # المحاولة الأولى: التحميل العادي (مناسب لملفات Keras 3)
            try:
                model = tf.keras.models.load_model(model_path, compile=False)
                return model, True
            except Exception:
                # المحاولة الثانية: إذا كان الملف SavedModel قديم، نستخدم TFSMLayer كما اقترح الخطأ
                st.info("التحميل بتنسيق TFSMLayer...")
                input_layer = keras.Input(shape=(512, 512, 1))
                # تأكد من أن المجلد يحتوي على ملفات pb إذا كنت تستخدم هذا المسار
                model_layer = keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
                outputs = model_layer(input_layer)
                model = keras.Model(input_layer, outputs)
                return model, True
                
        except Exception as e:
            st.error(f"⚠️ فشل التحميل النهائي: {e}")
            return None, False
    return None, False
