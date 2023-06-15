def retrieve_class_properties(class_: type):
    return [name for name, object in class_.__dict__.items() if isinstance(object, property)]