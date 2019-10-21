import numpy as np 
import shapely
from shapely.geometry import Polygon,MultiPoint  #多边形


def getIOU(line1, line2):

    a=np.array(line1).reshape(4, 2)   #四边形二维坐标表示
    poly1 = Polygon(a).convex_hull  #python四边形对象，会自动计算四个点，最后四个点顺序为：左上 左下  右下 右上 左上

    
    b=np.array(line2).reshape(4, 2)
    poly2 = Polygon(b).convex_hull

    
    union_poly = np.concatenate((a,b))   #合并两个box坐标，变为8*2

    if not poly1.intersects(poly2): #如果两四边形不相交
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area   #相交面积
            #union_area = poly1.area + poly2.area - inter_area
            union_area = MultiPoint(union_poly).convex_hull.area
            if union_area == 0:
                iou= 0
            #iou = float(inter_area) / (union_area-inter_area)  #错了
            iou=float(inter_area) / union_area
            # iou=float(inter_area) /(poly1.area+poly2.area-inter_area)
            # 源码中给出了两种IOU计算方式，第一种计算的是: 交集部分/包含两个四边形最小多边形的面积  
            # 第二种： 交集 / 并集（常见矩形框IOU计算方式） 
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    
    return iou

# line1=[908,215,934,312,752,355,728,252]   #四边形四个点坐标的一维数组表示，[x,y,x,y....]
# line2=[923,308,758,342,741,262,907,228]

# line1 = [0.168791, 0.193660, 0.921491, 0.339100, 0.816583, 0.792812, 0.097121, 0.653795]
# line2 = [0.770978, 0.980830, 0.971217, 0.996977, 0.950981, 0.820186, 0.838455, 0.983246]

# line1 = [.5, .5, .5, .6,  .6, .6, .6, .5]
# line2 = [.55, .55, .55, .65, .65, .65, .65, .55]

# print(getIOU(line1, line2))