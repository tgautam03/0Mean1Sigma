import math
from manim import *

MY_GOLD = "#DAA520"

def remove_invisible_chars(mobject: SVGMobject) -> SVGMobject:
    """Function to remove unwanted invisible characters from some mobjects.

    Parameters
    ----------
    mobject
        Any SVGMobject from which we want to remove unwanted invisible characters.

    Returns
    -------
    :class:`~.SVGMobject`
        The SVGMobject without unwanted invisible characters.
    """
    # TODO: Refactor needed
    iscode = False
    if mobject.__class__.__name__ == "Text":
        mobject = mobject[:]
    elif mobject.__class__.__name__ == "Code":
        iscode = True
        code = mobject
        mobject = mobject.code
    mobject_without_dots = VGroup()
    if mobject[0].__class__ == VGroup:
        for i in range(len(mobject)):
            mobject_without_dots.add(VGroup())
            mobject_without_dots[i].add(*(k for k in mobject[i] if k.__class__ != Dot))
    else:
        mobject_without_dots.add(*(k for k in mobject if k.__class__ != Dot))
    if iscode:
        code.code = mobject_without_dots
        return code
    return mobject_without_dots

class Simulation_3D(Scene):
    def construct(self):
        # Opening
        opening = Tex(r"Chapter 5: 3D Simulation", color=MY_GOLD).scale(1.25)
        self.play(FadeIn(opening))
        self.wait(1)
        self.play(FadeOut(opening))
        self.wait(2)

        # 3D Wave Equation
        wave_eq = Tex(r"$\frac{\partial^2 u(x,y,z,t)}{\partial t^2}$", r"$=$", r"$c^2(x,y,z)$", 
                      r"$\bigg[\frac{\partial^2 u(x,y,z,t)}{\partial x^2}+\frac{\partial^2 u(x,y,z,t)}{\partial y^2} + \frac{\partial^2 u(x,y,z,t)}{\partial z^2}\bigg]$", 
                      r"$+$", r"$s(x,y,z,t)$", color=WHITE).scale(0.85).to_edge(UP).shift(0.25*DOWN)
        self.play(Write(wave_eq))
        self.wait(2)

        # FadeOut
        self.play(
            *[FadeOut(mob)for mob in self.mobjects]
        )

        self.wait(2)