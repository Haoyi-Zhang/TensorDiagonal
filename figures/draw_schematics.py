"""Vector drawing primitives; copied into each independent project generator."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Circle, FancyArrowPatch

plt.rcParams.update({'font.family':'STIXGeneral', 'mathtext.fontset':'stix',
                     'font.size':9, 'pdf.fonttype':42, 'ps.fonttype':42,
                     'text.usetex':False, 'axes.unicode_minus':False})
CM = 1 / 2.54

class Drawing:
    def __init__(self, bounds):
        x0,y0,x1,y1 = bounds
        self.fig = plt.figure(figsize=((x1-x0)*CM,(y1-y0)*CM))
        self.ax = self.fig.add_axes([0,0,1,1], xlim=(x0,x1), ylim=(y0,y1))
        self.ax.set_aspect('equal'); self.ax.axis('off')
        self.nodes = {}
        self.labels = []

    def text(self,x,y,label,size=9,ha='center',va='center'):
        artist=self.ax.text(x,y,label,fontsize=size,ha=ha,va=va,
                            linespacing=1.15,clip_on=False,zorder=4)
        self.labels.append([artist,None])
        return artist

    def box(self,name,x,y,w,h,label,dashed=False,shape='rect',size=9):
        style = {'edgecolor':'black','facecolor':'white','linewidth':.55,
                 'linestyle':(0,(4,3)) if dashed else 'solid', 'zorder':2}
        if shape == 'diamond':
            obj = Polygon([(x-w/2,y),(x,y+h/2),(x+w/2,y),(x,y-h/2)],**style)
        elif shape == 'circle': obj = Circle((x,y),w/2,**style)
        else: obj = Rectangle((x-w/2,y-h/2),w,h,**style)
        self.ax.add_patch(obj); self.text(x,y,label,size)
        self.labels[-1][1]=(x-w/2,y-h/2,x+w/2,y+h/2)
        self.nodes[name] = (x,y,w,h,shape)

    def point(self,name,side):
        x,y,w,h,_ = self.nodes[name]
        return {'n':(x,y+h/2),'s':(x,y-h/2),'e':(x+w/2,y),
                'w':(x-w/2,y),'nw':(x-w/2,y+h/2),
                'ne':(x+w/2,y+h/2),'c':(x,y)}[side]

    def line(self,points,dashed=False):
        xs,ys=zip(*points)
        self.ax.plot(xs,ys,color='black',linewidth=.55,
                     linestyle=(0,(4,3)) if dashed else 'solid',zorder=1)

    def arrow(self,p,q,dashed=False,via=()):
        points=[p,*via,q]
        if via:self.line(points[:-1],dashed)
        self.ax.add_patch(FancyArrowPatch(points[-2],points[-1],
                         arrowstyle='-|>',mutation_scale=7.5,linewidth=.55,
                         color='black',shrinkA=0,shrinkB=0,
                         linestyle=(0,(4,3)) if dashed else 'solid',zorder=1))

    def edge(self,a,b,sa='s',sb='n',**kwargs):
        self.arrow(self.point(a,sa),self.point(b,sb),**kwargs)

    def frame(self,x,y,w,h):
        self.ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor='black',
                                   linewidth=.55,linestyle=(0,(4,3)),zorder=0))

    def save(self,out,name):
        # Fixed physical canvas: do not enlarge or rasterize the scientific figure.
        self.fig.canvas.draw()
        renderer=self.fig.canvas.get_renderer()
        x0,x1=self.ax.get_xlim();y0,y1=self.ax.get_ylim()
        for artist,node in self.labels:
            b=artist.get_window_extent(renderer).transformed(self.ax.transData.inverted())
            if b.x0<x0-.02 or b.x1>x1+.02 or b.y0<y0-.02 or b.y1>y1+.02:
                raise ValueError(name+': clipped label '+repr(artist.get_text()))
            if node and (b.x0<node[0]-.02 or b.x1>node[2]+.02 or b.y0<node[1]-.02 or b.y1>node[3]+.02):
                raise ValueError(name+': label exceeds node '+repr(artist.get_text()))
        self.fig.savefig(out/(name+'.pdf'),format='pdf',
                         metadata={'Creator':'Python Matplotlib','Producer':None,
                                   'CreationDate':None,'ModDate':None})
        plt.close(self.fig)






ROOT = Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description="Render ambiguity components and anchor pins; no experiment.")
    parser.add_argument("--out",type=Path,default=ROOT/"figures")
    parser.add_argument("--paper-dir",type=Path,default=None)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    d=Drawing((-.6,-1.5,10.3,2.25))
    for prefix,offset in (("a",0),("b",5.2)):
        for k,x,y in ((1,0,1.2),(2,2.1,1.2),(3,1.05,-.45)):
            d.box(prefix+str(k),x+offset,y,.6,.6,str(k),shape="circle",size=9)
        d.line([d.point(prefix+"1","e"),d.point(prefix+"2","w")])
        d.line([d.point(prefix+"1","s"),d.point(prefix+"3","nw")])
        d.line([d.point(prefix+"2","s"),d.point(prefix+"3","ne")])
        d.text(offset+1.05,1.52,r"$g_{12}=1$",size=9)
        d.text(offset+.3,.35,"1",size=9)
        d.text(offset+1.8,.35,"1",size=9)
    d.box("pin",9.3,.45,.6,.6,"4",size=9)
    d.arrow(d.point("b2","e"),d.point("pin","w"),dashed=True)
    d.arrow(d.point("b3","e"),d.point("pin","w"),dashed=True)
    d.arrow(d.point("b1","n"),d.point("pin","n"),dashed=True,
            via=((5.2,1.9),(9.3,1.9)))
    # Distinguish one-sided pin information from solid reciprocal-gain edges.
    for line in d.ax.lines[-2:]:
        if line.get_linestyle()!="-" : line.set_linestyle((0,(1,2)))
    for artist in d.ax.patches:
        if isinstance(artist,FancyArrowPatch):artist.set_linestyle((0,(1,2)))
    d.text(1.05,-1.2,r"$(a)\quad z=(1,1,1)$",size=9)
    d.text(6.7,-1.2,r"$(b)\quad z=(1,1,1,0)$",size=9)
    d.text(9.5,-.6,"pinned\n"+r"$z_4=0$",size=9)
    d.save(args.out,"component-structure")
    if args.paper_dir is not None:
        import shutil
        args.paper_dir.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(args.out/"component-structure.pdf",args.paper_dir/"component-structure.pdf")
    print("Rendered reciprocal gains and one-sided pinning without changing the graph.")

if __name__=="__main__":
    main()
