import json
import logging
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from .utils import AdvancedChatbot

logger = logging.getLogger(__name__)

# Initialize chatbot
try:
    print("Attempting to initialize chatbot...")
    KNOWLEDGE_BASE_PATH = 'knowledge_base.json'
    chatbot = AdvancedChatbot(knowledge_base_path=KNOWLEDGE_BASE_PATH)
    logger.info("Chatbot initialized")
except Exception as e:
    chatbot = None
    logger.error("Chatbot initialization failed!", exc_info=True)
    print("Chatbot init error:", str(e))
    

pending_save = {}

@csrf_exempt
def chat_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST requests only"}, status=405)

    if not chatbot:
        return JsonResponse({"error": "Chatbot unavailable"}, status=503)

    try:
        data = json.loads(request.body)
        user_message = data.get("message", "").strip()
        if len(user_message) < 2:
            return JsonResponse({"error": "Message too short"}, status=400)

        history = data.get("history", [])[:5]
        response_text, timings, retrieved_docs, should_save = chatbot.generate_response(
            user_message, history=history
        )

        if should_save:
            pending_save[user_message] = response_text

        return JsonResponse({
            "response": response_text,
            "timings": timings,
            "context": retrieved_docs
        })
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return JsonResponse({"error": "Server error"}, status=500)


@csrf_exempt
def feedback_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    try:
        data = json.loads(request.body)
        helpful = data.get("helpful", None)

        if helpful is True:
            # Save all pending Q&A
            for question, answer in pending_save.items():
                chatbot.retriever.add_document(question, answer)
            pending_save.clear()
            logger.info("Feedback positive — saved knowledge.")
        else:
            pending_save.clear()
            logger.info("Feedback negative — discarded knowledge.")

        return JsonResponse({"status": "success"})
    except Exception as e:
        logger.error(f"Feedback API error: {str(e)}")
        return JsonResponse({"error": "Invalid data"}, status=400)


def chat_ui(request):
    return render(request, 'chat.html', {
        'chatbot_name': chatbot.chatbot_name if chatbot else "Chatbot"
    })
