from mpl_toolkits import mplot3d
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
from sklearn.preprocessing import StandardScaler

meats = ['diezmillo3' , 'diezmillo4' ,  'diezmillo5','sirloin3','sirloin5']
#meats = ['diezmillo1','diezmillo2','diezmillo3','diezmillo4','diezmillo5','sirloin1','sirloin2','sirloin3','sirloin4','sirloin5']

days = [ 'dia1' , 'dia3' , 'dia4' , 'dia5' , 'dia6' ,  'dia7' ,  'dia8' ,  'dia9' , ]
days = [ 'dia1' , 'dia5' , 'dia9' , ]

def get_img_name( meat_id , day_id ) :
    meat = meats[ meat_id ]
    day = days[ day_id ]
    file_name = f'Carnes_Clau/{meat}_{day}.jpeg'
    return file_name

def get_img( img_name ) :
    img = cv2.imread( img_name )
    img = cv2.cvtColor( img , cv2.COLOR_BGR2RGB,img )

    return img

def show_all_imgs( ) :
    for day_id in range( len( days ) ) : 
        for meat_id in range( len( meats ) ) : 
            img_name = get_img_name( meat_id , day_id ) 
            img = get_img( img_name ) 
            plt.subplot( len( meats ) , len( days ) , 1 + meat_id * len( days ) + day_id ) 
            plt.imshow( img )
            plt.xticks([])
            plt.yticks([])
    plt.show( ) 

def map_hue( hue ) :
    # Red toned hues vary from 0 to 20 and 160 to 180
    # For better manageability apply a shift of +20 
    # Now red varied from 0 to 40
    return ( hue + 20 ) % 180

def get_hsv_rgb_sample(img,n_samples):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv = np.reshape(hsv, (hsv.shape[0] * hsv.shape[1], hsv.shape[2]))
    img = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
    idx = hsv[:, 1] > 0
    hsv = hsv[idx]
    img = img[idx]
    idx = np.random.permutation(len(hsv))
    idx = idx[:n_samples]
    hsv = hsv[idx]
    img = img[idx]
    return hsv,img

def get_lab_rgb_sample(img,n_samples):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    lab = np.reshape(lab, (lab.shape[0] * lab.shape[1], lab.shape[2]))
    img = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
    idx = lab[:, 0] > 0
    lab = lab[idx]
    img = img[idx]
    idx = np.random.permutation(len(lab))
    idx = idx[:n_samples]
    lab = lab[idx]
    img = img[idx]
    return lab,img

def color_sampling3d( img , ax ) :
    lab,img = get_lab_rgb_sample(img,2000)
    ax.scatter( lab[:,0] , lab[:,1] , lab[:,2] , c = img/255 ,alpha = 0.1)
    ax.set_xlim([0,255])
    ax.set_ylim([100,200])
    ax.set_zlim([100, 200])
    ax.set_xlabel('L')
    ax.set_ylabel('a')
    ax.set_zlabel('b')

def color_sampling2d(img,ax,channels):
    lab, img = get_lab_rgb_sample(img, 100)
    if channels=='La':
        ax.scatter(lab[:,0] , lab[:,1], c=img/255, alpha=0.1)
        ax.set_xlim([0,255])
        ax.set_ylim([100,200])
        ax.set_xlabel('L')
        ax.set_ylabel('a')
    elif channels=='Lb':
        ax.scatter(lab[:,0] , lab[:,2], c=img/255, alpha=0.1)
        ax.set_xlim([0,255])
        ax.set_ylim([100,200])
        ax.set_xlabel('L')
        ax.set_ylabel('b')
    elif channels=='ab':
        ax.scatter(lab[:,1] , lab[:,2], c=img/255,alpha=0.1)
        ax.set_xlim([100,200])
        ax.set_ylim([100,200])
        ax.set_xlabel('a')
        ax.set_ylabel('b')
    ax.set_xticks([])
    ax.set_yticks([])

def show_color_sampling( ) :
    img_name = 'Carnes_Clau/diezmillo3_dia1.jpeg'
    img = get_img( img_name )
    fig = plt.figure( )
    ax = fig.add_subplot(1,2,1)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    ax = fig.add_subplot( 1 , 2, 2, projection = '3d' )
    plt.xticks([])
    plt.yticks([])
    color_sampling3d( img , ax )
    plt.show( )

def get_reds( img ) : 
    hsv = cv2.cvtColor( img , cv2.COLOR_RGB2HSV ) 
    msk = np.bitwise_or( hsv[ :  , :  , 0 ] < 20 , hsv[ :  , :  , 0 ] > 160 ) 
    return msk

def blackout( img , msk ) : 
    hsv = cv2.cvtColor( img , cv2.COLOR_RGB2HSV ) 
    h , s , v = cv2.split( hsv ) 
    v[ msk ] = 0
    hsv = cv2.merge(( h , s , v ) ) 
    img = cv2.cvtColor( hsv , cv2.COLOR_HSV2RGB ) 
    return img

def show_reds( ) :
    img_name = 'Carnes_Clau/diezmillo3_dia3.jpeg'
    img = get_img( img_name )
    reds = get_reds( img )
    img = blackout( img, ~reds )
    plt.imshow( img )
    plt.xticks([])
    plt.yticks([])
    plt.show( )

def get_largest_component( msk ) : 
    n_components , labels , stats , _ = cv2.connectedComponentsWithStats( msk.astype( np.uint8 ) , connectivity = 4 ) 
    max_area = 0
    for i in range( 1 , n_components ) : 
        area = stats[ i , cv2.CC_STAT_AREA ] 
        if area > max_area:
            max_area = area
            msk = labels == i
    return msk

def show_meat( ) :
    img_name = 'Carnes_Clau/diezmillo3_dia1.jpeg'
    img = get_img( img_name )
    reds = get_reds( img )
    meat = get_largest_component( reds )
    img = blackout( img, ~meat )
    plt.imshow( img )
    plt.xticks([])
    plt.yticks([])
    plt.show( )

def show_all_meats( ) :
    for day_id in range( len( days ) ) : 
        for meat_id in range( len( meats ) ) : 
            img_name = get_img_name( meat_id , day_id ) 
            img = get_img( img_name ) 
            reds = get_reds( img ) 
            meat = get_largest_component( reds ) 
            img = blackout( img , ~meat ) 
            plt.subplot( len( meats ) , len( days ) , 1 + meat_id * len( days ) + day_id ) 
            plt.imshow( img )
            plt.xticks([])
            plt.yticks([])
    plt.show( )

def show_color_sampling_changes_3d(meat_n):
    fig = plt.figure( )
    for meat_id in [ meat_n ]:
        for day_id in range( len( days ) ) :
            img_name = get_img_name ( meat_id , day_id )
            img = get_img( img_name )
            reds = get_reds( img )
            meat = get_largest_component( reds )
            img = blackout( img , ~meat )

            fig.add_subplot(2, len(days), 1 + day_id)
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])

            ax = fig.add_subplot( 2 , len( days ) , 1 + day_id + len(days) , projection = '3d' )
            color_sampling3d( img , ax )
    plt.show( )

def show_color_sampling_changes_2d(meat_n ) :
    fig = plt.figure( )
    for meat_id in [ meat_n ]:
        for day_id in range( len( days ) ) : 
            img_name = get_img_name ( meat_id , day_id )
            img = get_img( img_name )
            reds = get_reds( img )
            meat = get_largest_component( reds )
            img = blackout( img , ~meat )

            fig.add_subplot(4, len(days), 1 + day_id )
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])

            ax = fig.add_subplot(4,len( days ) , 1 + day_id + len(days) )
            color_sampling2d(img,ax,'La')

            ax = fig.add_subplot(4, len(days), 1 + day_id + len(days)*2)
            color_sampling2d(img, ax, 'Lb')

            ax = fig.add_subplot(4, len(days), 1 + day_id + len(days)*3)
            color_sampling2d(img, ax, 'ab')

    plt.show( )

def show_color_sampling_changes_1d(meat_n ) :
    fig = plt.figure()
    for meat_id in [meat_n]:
        H = []
        S = []
        V = []
        for day_id in range(len(days)):
            img_name = get_img_name(meat_id, day_id)
            img = get_img(img_name)
            reds = get_reds(img)
            meat = get_largest_component(reds)
            img = blackout(img, ~meat)

            fig.add_subplot(4, len(days), 1 + day_id)
            plt.imshow(img)
            plt.title('Day '+str(days[day_id][-1]))
            plt.xticks([])
            plt.yticks([])

            hsv,rgb = get_lab_rgb_sample(img,500)
            H.append(hsv[:,0])
            S.append(hsv[:,1])
            V.append(hsv[:,2])

        H = np.array(H).transpose()
        V = np.array(V).transpose()
        S = np.array(S).transpose()

        fig.add_subplot(4, 1, 2)
        plt.boxplot(H)
        plt.ylabel('L')
        plt.xticks([])
        fig.add_subplot(4, 1, 3)
        plt.boxplot(V)
        plt.ylabel('a')
        plt.xticks([])
        fig.add_subplot(4, 1, 4)
        plt.boxplot(S)
        plt.ylabel('b')
        plt.xticks([])
    plt.suptitle('Meat '+meats[meat_id])
    plt.show()

def show_color_sampling_changes_1d_histograms(meat_n) :
    fig = plt.figure()
    for meat_id in [meat_n]:
        for day_id in range(len(days)):
            img_name = get_img_name(meat_id, day_id)
            img = get_img(img_name)
            reds = get_reds(img)
            meat = get_largest_component(reds)
            img = blackout(img, ~meat)

            fig.add_subplot(4, len(days), 1 + day_id)
            plt.imshow(img)
            plt.title('Day '+str(days[day_id][-1]))
            plt.xticks([])
            plt.yticks([])

            hsv, rgb = get_lab_rgb_sample(img, 500)
            fig.add_subplot(4,len(days),1+day_id+len(days))
            plt.hist(hsv[:,0],range=[0,255])
            plt.ylim([0,300])
            plt.xticks([])
            if day_id > 0:
                plt.yticks([])
            else:
                plt.ylabel('L')

            fig.add_subplot(4, len(days), 1 + day_id + len(days)*2)
            plt.hist(hsv[:, 1], range=[0, 255])
            plt.ylim([0,300])
            if day_id > 0:
                plt.yticks([])
            else:
                plt.ylabel('a')
            plt.xticks([])

            fig.add_subplot(4, len(days), 1 + day_id + len(days)*3)
            plt.hist(hsv[:, 2], range=[0, 255])
            plt.ylim([0,300])
            if day_id > 0:
                plt.yticks([])
            else:
                plt.ylabel('b')
    plt.suptitle('Meat '+meats[meat_id])
    plt.show()

def show_color_mean_changes( ) :
    fig = plt.figure( )
    ax = fig.add_subplot( 1 , 1 , 1 , projection = '3d' )
    ax.set_xlabel( 'H' ) 
    ax.set_ylabel( 'S' ) 
    ax.set_zlabel( 'V' ) 
    ax.set_xlim( [ 0 , 256 ] )
    ax.set_ylim( [ 0 , 256 ] )
    ax.set_zlim( [ 0 , 256 ] )

    for meat_id in range( len( meats ) ):    
        h_list = [ ]
        s_list = [ ]
        v_list = [ ]
        for day_id in range( len( days ) ) : 
    
            img_name = get_img_name ( meat_id , day_id )
            img = get_img( img_name )
            reds = get_reds( img )
            meat = get_largest_component( reds )
    
            hsv = cv2.cvtColor( img , cv2.COLOR_RGB2HSV ) 
            h , s , v = cv2.split( hsv ) 
    
            h_mean = map_hue( h[ meat ] ).mean( )
            s_mean = s[ meat ].mean( )
            v_mean = v[ meat ].mean( )
    
            h_list.append( h_mean )
            s_list.append( s_mean )
            v_list.append( v_mean )
    
        ax.plot3D( h_list , s_list , v_list )

    plt.show( )

def eigendecomposition( X , ax , c) :
    X_std = X - X.mean( axis = 0 )

    cov_mat = np.cov( X_std.T )
    eig_vals, eig_vecs = np.linalg.eig( cov_mat )
    
    h_mean,s_mean,v_mean = X.mean( axis = 0 )
    
    x = h_mean * np.ones( 3 ) # x coordinates of origin
    y = s_mean * np.ones( 3 ) # y coordinates of origin
    z = v_mean * np.ones( 3 ) # z coordinates of origin
    u = eig_vecs[ 0 ] * np.sqrt( eig_vals ) * 2 # x coordinates of vectors
    v = eig_vecs[ 1 ] * np.sqrt( eig_vals ) * 2 # y coordinates of vectors
    w = eig_vecs[ 2 ] * np.sqrt( eig_vals ) * 2 # z coordinates of vectors

    ax.quiver( x , y , z , u , v , w , colors = c )

def show_eigendecomposition( ):
    fig=plt.figure()
    ax = fig.add_subplot( 1 , 1, 1, projection = '3d' )
    ax.set_xlabel( 'H' ) 
    ax.set_ylabel( 'S' ) 
    ax.set_zlabel( 'V' ) 
    ax.set_xlim( [ 0 , 256 ] )
    ax.set_ylim( [ 0 , 256 ] )
    ax.set_zlim( [ 0 , 256 ] )
    
    img_name = get_img_name( 0 , 0 )
    img = get_img( img_name )
    reds = get_reds( img )
    meat = get_largest_component( reds )
    img = blackout( img , ~meat )
    
    hsv = cv2.cvtColor( img , cv2.COLOR_RGB2HSV ) 
    X = hsv[ meat ].reshape( ( -1 , 3 ) )
    X[:,0] = map_hue( X[ : , 0 ] )
            
    eigendecomposition( X , ax , 'b' )

    color_sampling( img , ax )

    plt.show( )


def show_eigendecomposition_changes( ):
    fig=plt.figure()
    c = 'bgrcmy'
    ax = fig.add_subplot( 1 , 1, 1, projection = '3d' )
    ax.set_xlabel( 'H' ) 
    ax.set_ylabel( 'S' ) 
    ax.set_zlabel( 'V' ) 
    ax.set_xlim( [ 0 , 256 ] )
    ax.set_ylim( [ 0 , 256 ] )
    ax.set_zlim( [ 0 , 256 ] )

    for meat_id in range( len( meats ) ):
        for day_id in range( len( days ) ):
            img_name = get_img_name( meat_id , day_id )
            img = get_img( img_name )
            reds = get_reds( img )
            meat = get_largest_component( reds )
            img = blackout( img , ~meat )
    
            hsv = cv2.cvtColor( img , cv2.COLOR_RGB2HSV ) 
            X = hsv[ meat ].reshape( ( -1 , 3 ) )
            X[:,0] = map_hue( X[ : , 0 ] )
            
            eigendecomposition( X , ax , c[ day_id ] )
        
    plt.show( )

def calculate_matrix_values():
    L = np.zeros((len(meats),len(days)))
    a = np.zeros((len(meats),len(days)))
    b = np.zeros((len(meats),len(days)))
    for meat_id in range( len( meats ) ):
        for day_id in range( len( days ) ):
            img_name = get_img_name( meat_id , day_id )
            img = get_img(img_name)
            lab,rgb = get_lab_rgb_sample(img,100)
            L[meat_id,day_id] = np.mean(lab[:,0])
            a[meat_id,day_id] = np.mean(lab[:, 1])
            b[meat_id,day_id] = np.mean(lab[:, 2])
    dfl = pandas.DataFrame(L,index=meats,columns=days)
    dfa = pandas.DataFrame(a,index=meats,columns=days)
    dfb = pandas.DataFrame(b,index=meats,columns=days)
    print(dfl)
    print(dfa)
    print(dfb)

def kullback_leibler(P,Q):
    P = P/np.sum(P)
    Q = Q/np.sum(Q)
    bins = len(P)
    kl = 0
    for b in range(bins):
        if Q[b] != 0 and P[b] !=0 :
            kl += P[b]*abs(np.log(P[b]/Q[b]))
    if np.isnan(kl):
        print('kl',kl)
        print('P',P)
        print('Q',Q)
    return kl

def calculate_kl_distances():
    L = np.zeros((len(meats), len(days)))
    a = np.zeros((len(meats), len(days)))
    b = np.zeros((len(meats), len(days)))
    KLlab = np.zeros((len(meats), len(days)))
    KLab = np.zeros((len(meats), len(days)))
    Euclidean = np.zeros((len(meats), len(days)))
    for meat_id in range( len( meats ) ):
        for day_id in range(len(days)):
            img_name = get_img_name(meat_id, day_id)
            img = get_img(img_name)
            reds = get_reds(img)
            meat = get_largest_component(reds)
            img = blackout(img, ~meat)
            lab, rgb = get_lab_rgb_sample(img, 500)
            if day_id == 0:
                Pl,_ = np.histogram(lab[:,0],bins=10,range=[0, 255])
                Pa,_ = np.histogram(lab[:,1],bins=10,range=[0, 255])
                Pb,_ = np.histogram(lab[:,2],bins=10,range=[0, 255])
                Pml = np.mean(lab[:,0])
                Pma = np.mean(lab[:,1])
                Pmb = np.mean(lab[:,2])
            else:
                Ql,_ = np.histogram(lab[:,0],bins=10,range=[0, 255])
                Qa,_ = np.histogram(lab[:,1],bins=10,range=[0, 255])
                Qb,_ = np.histogram(lab[:,2],bins=10,range=[0, 255])
                Qml = np.mean(lab[:, 0])
                Qma = np.mean(lab[:, 1])
                Qmb = np.mean(lab[:, 2])
                L[meat_id,day_id] = (kullback_leibler(Pl,Ql)+kullback_leibler(Ql,Pl))/2
                a[meat_id,day_id] = (kullback_leibler(Pa,Qa)+kullback_leibler(Qa,Pa))/2
                b[meat_id,day_id] = (kullback_leibler(Pb,Qb)+kullback_leibler(Qb,Pb))/2
                Euclidean[meat_id,day_id] = math.sqrt( (Pml-Qml)**2+(Pma-Qma)**2+(Pmb-Qmb)**2 )
    KLlab = (L+a+b)/3
    KLab = (a+b)/2
    dfl = pandas.DataFrame(L, index=meats, columns=days)
    dfa = pandas.DataFrame(a, index=meats, columns=days)
    dfb = pandas.DataFrame(b, index=meats, columns=days)
    dfKLlab = pandas.DataFrame(KLlab, index=meats, columns=days)
    dfKLab = pandas.DataFrame(KLab, index=meats, columns=days)
    dfEuc = pandas.DataFrame(Euclidean, index=meats, columns=days)
    print('Euclidean')
    print(dfEuc)
    print('Kullback Leibler LAB')
    print(dfKLlab)


#show_all_imgs( )
#show_color_sampling( )
#show_reds( )
#show_meat( )
#show_all_meats( )

calculate_kl_distances()
show_color_sampling_changes_1d_histograms(0)
#show_color_sampling_changes_1d(0)
#show_color_sampling_changes_1d(1)
#show_color_sampling_changes_1d(2)
#show_color_sampling_changes_1d(3)
#show_color_sampling_changes_1d(4)
#show_color_sampling_changes_2d(0)
#show_color_sampling_changes_3d(0)
calculate_matrix_values()
'''
# show_color_mean_changes( )
#show_eigendecomposition( )
#show_eigendecomposition_changes( )
'''
