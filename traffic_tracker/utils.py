from deep_sort_realtime.deepsort_tracker import DeepSort


def get_tracker():
    
    
    return DeepSort(
        max_age = 20,
        n_init = 3,
        nms_max_overlap = 0.7,
        max_cosine_distance = 0.4,
        nn_budget = None, 
        override_track_class = None
    )


