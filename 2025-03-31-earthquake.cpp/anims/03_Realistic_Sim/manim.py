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

class Realistic_Sim(Scene):
    def construct(self):
        # Opening
        opening = Tex(r"Chapter 3: Realistic Simulation", color=MY_GOLD).scale(1.5)
        self.play(FadeIn(opening))
        self.wait(1)
        self.play(FadeOut(opening))
        self.wait(2)

        # Problem in the simulation
        window_1 = Square(side_length=4, color=MY_GOLD, stroke_width=10).shift(2.25*LEFT+DOWN)
        window_2 = Square(side_length=4, color=MY_GOLD, stroke_width=10).shift(2.25*RIGHT+DOWN)

        arrow = CurvedArrow(window_1.get_edge_center(UP), window_2.get_edge_center(UP), color=MY_GOLD, angle=-PI / 2)
        comment = Tex(r"Displacement at the surface", color=MY_GOLD).next_to(arrow, UP)

        self.play(Create(arrow), Write(comment))
        self.wait(2)

        self.play(FadeOut(arrow, comment))

        # Damping boundary
        title = Title("Realistic Surface")
        self.play(Write(title))
        self.wait(2)

        damp_bc = Tex(r"$\frac{\partial u(y,t)}{\partial t} + c(y) \frac{\partial u}{\partial n} = -\alpha\frac{\partial u}{\partial t}$").shift(UP)
        self.play(Write(damp_bc))
        self.wait(1)

        damp_bc_surf = Tex(r"$\frac{u(0,t+dt)-u(0,t)}{dt} + c(0) \frac{u(0+dy,t)-u(0,t)}{dy} = -\alpha \frac{u(0,t)-u(0,t-dt)}{dt}$").next_to(damp_bc, DOWN).shift(0.5*DOWN)
        self.play(Write(damp_bc_surf))
        self.wait(1)

        damp_bc_surf_exp = Tex(r"$u(0,t+dt) = u(0,t) + c(0) \frac{dt}{dy} (u(0+dy,t)-u(0,t))  -\alpha(u(0,t)-u(0,t-dt))$").scale(0.75).move_to(damp_bc_surf)
        self.play(ReplacementTransform(damp_bc_surf, damp_bc_surf_exp))
        self.wait(1)

        self.play(FadeOut(damp_bc), damp_bc_surf_exp.animate.shift(2.5*UP))
        self.wait(2)

        # FadeOut
        self.play(
            *[FadeOut(mob)for mob in self.mobjects]
        )

        # Absorbing boundary
        title = Title("Infinite Domain")
        self.play(Write(title))
        self.wait(2)

        mur_bc = Tex(r"$u(ny-1,t+dt) = u(ny-2,t) + \frac{c(ny-1) \cdot dt - dx}{c(nx-1) \cdot dt + dx} \bigg(u(ny-2,t+dt) - u(ny-1,t)\bigg)$").scale(0.7)

        self.play(Write(mur_bc))
        self.wait(1)
        self.play(mur_bc.animate.shift(2*UP))
        self.wait(2)

        # FadeOut
        self.play(
            *[FadeOut(mob)for mob in self.mobjects]
        )

        self.wait(2)