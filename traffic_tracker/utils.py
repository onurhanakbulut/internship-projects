from deep_sort_realtime.deepsort_tracker import DeepSort


def get_tracker():
    
    
    return DeepSort(
        max_age = 15,
        n_init = 2,
        nms_max_overlap = 0.6,
        max_cosine_distance = 0.2,
        nn_budget = 80, 
        override_track_class = None,
        
        
        embedder = 'mobilenet',
        embedder_gpu=True
    )


