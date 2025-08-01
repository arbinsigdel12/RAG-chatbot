# RAG-chatbot
chatbot with Retrieval-Augmented Generation (RAG) and Google Search integration.

# 🧠 Chatbot Model Comparison Table
### These are all the models I tested with

| Model Name                          | Type / Size          | ✅ Advantages                                                                 | ❌ Disadvantages                                                            | RAG Suitability | GPU Feasibility |
|-------------------------------------|-----------------------|------------------------------------------------------------------------------|-----------------------------------------------------------------------------|------------------|------------------|
| `GPT2LMHeadModel`                   | Decoder-only (117M)   | - Very lightweight<br>- Fast inference                                       | - No instruction tuning<br>- Poor coherence<br>- No factual awareness      | ❌ Weak           | ✅ Very Easy      |
| `google/flan-t5-small`             | Encoder-Decoder (80M) | - Instruction-tuned<br>- Good few-shot performance for size                 | - Lacks deep reasoning<br>- Short context window                           | ⚠️ Basic          | ✅ Very Easy      |
| `microsoft/DialoGPT-medium`        | Decoder-only (345M)   | - Fine-tuned for dialogue<br>- Fast responses                               | - Always casual tone<br>- Not instruction-tuned<br>- Struggles with facts  | ❌ Weak           | ✅ Easy           |
| `facebook/blenderbot-400M-distill` | Seq2Seq (400M)        | - Open-domain dialog model<br>- Fast<br>- Good at casual talk               | - Not fact-grounded<br>- Repetitive and non-specific answers               | ❌ Weak           | ✅ Easy           |
| `google/flan-t5-large`             | Encoder-Decoder (780M)| - Very capable instruction-following<br>- Handles diverse prompts            | - Limited factual grounding<br>- Lacks memory or doc awareness             | ⚠️ Medium         | ✅ OK (slow)      |
| `google/flan-t5-xl`                | Encoder-Decoder (3B)  | - Better than large at zero-shot tasks<br>- Fluent outputs                  | - Heavy for 3050 Ti<br>- Requires >12GB VRAM or CPU offload                | ⚠️ Medium         | ❌ Challenging    |
| `tiiuae/falcon-rw-1b`              | Decoder-only (1.3B)   | - Lightweight Falcon<br>- Some instruction tuning                           | - Repetition issues<br>- Poor prompt alignment<br>- Shallow answers        | ⚠️ Weak           | ✅ Good           |
| `tiiuae/falcon-7b-instruct`        | Decoder-only (7B)     | - Stronger instruction tuning<br>- Good context handling                     | - Disk offload needed<br>- High VRAM use<br>- Crashes w/o tuning           | ✅ Good           | ⚠️ Hard (offload) |
| `mistralai/Mistral-7B-Instruct-v0.2`| Decoder-only (7B)     | - **Excellent instruction following**<br>- **Fast on RTX 3050Ti**<br>- Long context<br>- FlashAttention supported | - Large download (~13GB)<br>- Needs quantization or `bfloat16`             | ✅ **Best**       | ✅ **Efficient**  |

---

### ✅ Summary

- **Best Overall**: `mistralai/Mistral-7B-Instruct-v0.2` – Excellent performance with RAG and fast on 3050 Ti.
- **Best Lightweight**: `tiiuae/falcon-rw-1b` – Good fallback if memory is tight.
- **Avoid**: `GPT2LMHeadModel`, `DialoGPT`, `Blenderbot` – Poor performance in modern chatbot/RAG setups.

## 🚀 How to Run the Chatbot Project Locally

Follow these steps to set up and run the chatbot on your local machine.

### 1. Clone the Repository
git clone https://github.com/arbinsigdel12/deep-chatbot.git<br>
cd deep-chatbot

### 2. 🛠️ Create and Activate a Virtual Environment
> **Note:** For Windows.

python -m venv venv  
venv\Scripts\activate  
pip install -r requirements.txt  

python manage.py runserver


