
class WindowBox:

    @staticmethod
    def is_acceptable_size(box):
        min_tolerance = 10
        max_tolerance = 150            
        if abs(box[1][0] - box[0][0]) >= min_tolerance and \
            abs(box[1][1] - box[0][1]) <= max_tolerance:
                return True
        return False
