import time

from app.base.exception.exception import show_log
from app.src.v1.schemas.base import LLMRequest, DoneLLMRequest, LoraTrainnerRequest, DoneLoraTrainnerRequest, UpdateStatusTaskRequest
from app.src.v1.backend.api import update_status_for_task, send_progress_task, send_done_llm_task
from app.services.ai_services.text_completion import run_text_completion_gemma_7b, run_gemma_finetuning
import os
from app.utils.services import minio_client


def finetune_gemma_7b(
    celery_task_id: str,
    request_data: LoraTrainnerRequest,
    input_paths: list[str],
    output_paths: list[str],
    minio_output_paths: list[str],
    
):
    show_log(
        message="function: lora_trainner, "
                f"celery_task_id: {celery_task_id}"
    )
    url_download = ''
    try :
        
        output_path = run_gemma_finetuning(
            data = request_data,
            
        )
        if not os.path.exists(output_path):
            is_success, response, error = update_status_for_task(
                UpdateStatusTaskRequest(
                    task_id=request_data['task_id'],
                    status="FAILED",
                    result=url_download,
                )
            )  
            return "Failed Lora Trainning"
        
    except Exception as e:
        print(e)
        return "Failed Lora Trainning"
    
    
    
    

def text_completion(
        celery_task_id: str,
        request_data: LLMRequest,
):
    show_log(
        message="function: text_completion "
                f"celery_task_id: {celery_task_id}"
    )
    try:
        response = run_text_completion_gemma_7b(request_data['messages'], request_data['config'])
        if not response:
            show_log(
                message="function: text_completion"
                        f"celery_task_id: {celery_task_id}, "
                        f"error: Update task status failed",
                level="error"
            )
            return response
        
        # send done task
        send_done_llm_task(
            request_data=DoneLLMRequest(
                task_id=request_data['task_id'],
                response=response
            )
        )

        return True, response, None
    except Exception as e:
        print(e)
        return False, None, str(e)
