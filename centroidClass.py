from collections import OrderedDict
from scipy.spatial import distance as dist

class TrackedObject:
    def __init__(self, objectID, centroid, maxHistory=5):
        # Assign an ID to the object and initialize properties
        self.objectID = objectID
        self.centroid = centroid
        self.centroid_history = [centroid]
        self.maxHistory = maxHistory
        self.disappeared = 0
        self.average_centroid = centroid

    def update_centroid(self, new_centroid):
        # Update the current centroid and maintain centroid history
        self.centroid = new_centroid
        self.centroid_history.append(new_centroid)
        if len(self.centroid_history) > self.maxHistory:
            self.centroid_history.pop(0)

        # Compute the average of the centroid history
        self.average_centroid = tuple(map(lambda x: int(sum(x) / len(x)), zip(*self.centroid_history)))

    def mark_disappeared(self):
        # Increase the disappeared count for the object
        self.disappeared += 1

    def reset_disappeared(self):
        # Reset disappeared count when object is visible again
        self.disappeared = 0


class CentroidTracker:
    def __init__(self, maxDisappeared=50, maxHistory=5):
        self.nextObjectID = 0
        self.objects = OrderedDict()  # Holds the TrackedObject instances
        self.maxDisappeared = maxDisappeared
        self.maxHistory = maxHistory

    def register(self, centroid):
        # Create a new tracked object and assign it the next available object ID
        obj = TrackedObject(self.nextObjectID, centroid, maxHistory=self.maxHistory)
        self.objects[self.nextObjectID] = obj
        self.nextObjectID += 1

    def deregister(self, objectID):
        # Remove the object from tracking
        del self.objects[objectID]

    def update(self, inputCentroids):
        # If no centroids are detected, mark existing objects as disappeared
        if len(inputCentroids) == 0:
            for objectID in list(self.objects.keys()):
                obj = self.objects[objectID]
                obj.mark_disappeared()

                if obj.disappeared > self.maxDisappeared:
                    self.deregister(objectID)

            return self.get_all_objects()

        # If no objects are currently tracked, register all centroids as new objects
        if len(self.objects) == 0:
            for centroid in inputCentroids:
                self.register(centroid)

        else:
            # Grab the current object IDs and their corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = [obj.centroid for obj in self.objects.values()]

            # Compute the distance between each pair of object centroids and input centroids
            distances = dist.cdist(objectCentroids, inputCentroids)

            # Find the smallest value in each row (min distances), and then sort rows by the min values
            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            # Update the objects with their new centroids
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                obj = self.objects[objectID]
                obj.update_centroid(inputCentroids[col])
                obj.reset_disappeared()

                usedRows.add(row)
                usedCols.add(col)

            # Register new input centroids as new objects
            for col in range(len(inputCentroids)):
                if col not in usedCols:
                    self.register(inputCentroids[col])

            # Mark remaining objects as disappeared
            for row in range(len(objectCentroids)):
                if row not in usedRows:
                    objectID = objectIDs[row]
                    obj = self.objects[objectID]
                    obj.mark_disappeared()

                    if obj.disappeared > self.maxDisappeared:
                        self.deregister(objectID)

        return self.get_all_objects()

    def get_all_objects(self):
        # Return current objects and their average centroids
        objects = {objectID: obj.centroid for objectID, obj in self.objects.items()}
        return objects
