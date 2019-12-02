import numpy as np 
import shapely
from shapely.geometry import Polygon,MultiPoint 



def getIOU(line1, line2):

    """
        give the two quadrilateral in the format (x1_q1, y1_q1, x2_q1, y2_q1, x3_q1, y3_q1, x4_q1, y4_q1) , (x1_q2, y1_q2, x2_q2, y2_q2, x3_q2, y3_q2, x4_q2, y4_q2)
        and it returns their IOU
    """

    a=np.array(line1).reshape(4, 2) 
    poly1 = Polygon(a).convex_hull 

    
    b=np.array(line2).reshape(4, 2)
    poly2 = Polygon(b).convex_hull

    
    union_poly = np.concatenate((a,b))  

    if not poly1.intersects(poly2):
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area
            union_area = MultiPoint(union_poly).convex_hull.area
            if union_area == 0:
                iou= 0
            iou=float(inter_area) / union_area

        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    
    return iou

# line1=[908,215,934,312,752,355,728,252] 
# line2=[923,308,758,342,741,262,907,228]

# line1 = [0.168791, 0.193660, 0.921491, 0.339100, 0.816583, 0.792812, 0.097121, 0.653795]
# line2 = [0.770978, 0.980830, 0.971217, 0.996977, 0.950981, 0.820186, 0.838455, 0.983246]

# line1 = [.5, .5, .5, .6,  .6, .6, .6, .5]
# line2 = [.55, .55, .55, .65, .65, .65, .65, .55]

# print(getIOU(line1, line2))