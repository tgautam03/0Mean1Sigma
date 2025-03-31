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

class Accuracy_Stability(Scene):
    def construct(self):
        # Opening
        opening = Tex(r"Chapter 4: Improving Accuracy and Stability", color=MY_GOLD).scale(1.25)
        self.play(FadeIn(opening))
        self.wait(1)
        self.play(FadeOut(opening))
        self.wait(2)

        # Problem in the simulation
        title = Title("Finite Difference Approximation", color=WHITE)

        self.play(Write(title))
        self.wait(2)

        # Higher order approximations
        f_2deriv = Tex(r"$f''(x) \approx \frac{f(x+dx)-2f(x)+f(x-dx)}{dx^2}$").scale(1).next_to(title, DOWN).shift(0.25*DOWN)

        self.play(Write(f_2deriv))
        self.wait(1)

        ax = Axes(x_range=[0, 1, 0.1],
                  y_range=[0, 10, 1],
                  tips=False,
                  axis_config={"stroke_width": 5,
                               "include_ticks": True,}).scale(0.6).shift(1*DOWN+ 4*RIGHT)
        x_label = ax.get_x_axis_label(Tex(r"Time (t) $\rightarrow$"), edge=DOWN, direction=DOWN, buff=0.5).shift(LEFT)
        y_label = ax.get_y_axis_label(Tex(r"$\leftarrow$ Depth (y)"), edge=LEFT, direction=LEFT, buff=0.5).rotate(90*DEGREES)
        self.play(FadeIn(y_label))

        dots_0_coarse = VGroup()
        ny = 11
        dy = 10/(ny-1)
        for i in range(ny):
            dots_0_coarse.add(Circle(radius=0.05, color=WHITE, fill_opacity=1).move_to(ax.c2p(0,i*dy)))

        self.play(FadeIn(dots_0_coarse))
        self.wait(1)

        box_dx = SurroundingRectangle(f_2deriv[0][:6], color=MY_GOLD, stroke_width=5)
        self.play(Create(box_dx))
        self.wait(1)
        circ_dx = Circle(radius=0.1, color=MY_GOLD).move_to(dots_0_coarse[5])
        self.play(ReplacementTransform(box_dx, circ_dx))
        self.wait(1)

        box_x_1 = SurroundingRectangle(f_2deriv[0][7:14], color=GREEN, buff=0.04, stroke_width=5)
        box_x = SurroundingRectangle(f_2deriv[0][16:20], color=BLUE, buff=0.04, stroke_width=5)
        box_x__1 = SurroundingRectangle(f_2deriv[0][21:28], color=RED, buff=0.04, stroke_width=5)
        self.play(Create(VGroup(box_x_1, box_x, box_x__1)))
        self.wait(1)

        circ_x_1 = Circle(radius=0.1, color=GREEN).move_to(dots_0_coarse[4])
        circ_x = Circle(radius=0.1, color=BLUE).move_to(dots_0_coarse[5])
        circ_x__1 = Circle(radius=0.1, color=RED).move_to(dots_0_coarse[6])
        self.play(ReplacementTransform(box_x_1, circ_x_1),ReplacementTransform(box_x, circ_x), ReplacementTransform(box_x__1, circ_x__1))
        self.wait(1)

        circ_x_3 = Circle(radius=0.1, color=GREEN_B).move_to(dots_0_coarse[2])
        circ_x_2 = Circle(radius=0.1, color=GREEN_A).move_to(dots_0_coarse[3])
        circ_x__2 = Circle(radius=0.1, color=RED_A).move_to(dots_0_coarse[7])
        circ_x__3 = Circle(radius=0.1, color=RED_B).move_to(dots_0_coarse[8])

        comment_1 = Tex("5 points", color=MY_GOLD).next_to(circ_x, RIGHT)
        self.play(FadeOut(f_2deriv), Create(VGroup(circ_x_2, circ_x__2)), Write(comment_1))
        self.wait(1)
        comment_2 = Tex("7 points", color=MY_GOLD).next_to(circ_x, RIGHT)
        self.play(Create(VGroup(circ_x_3, circ_x__3)), ReplacementTransform(comment_1, comment_2))
        self.wait(1)

        f_2deriv = Tex(r"$\frac{2f(x-3dx) - 27f(x-2dx) + 270f(x-dx) - 490f(x) + 270f(x+dx) - 27f(x+2dx) + 2f(x+3dx)}{180dx^2}$").scale(0.8).next_to(title, DOWN).shift(0.25*DOWN)

        self.play(Write(f_2deriv))
        self.wait(1)

        objects_to_keep = [title, f_2deriv]  # Replace with your actual objects
        self.play(
            *[FadeOut(mob) for mob in self.mobjects if mob not in objects_to_keep], 
        )
        self.wait(2)

        # FadeOut
        self.play(
            *[FadeOut(mob)for mob in self.mobjects]
        )

        self.wait(2)

        # CFL condition
        title = Title("Courant-Friedrichs-Lewy Condition")
        self.play(Write(title))

        ax = Axes(x_range=[0, 1, 0.1],
                  y_range=[0, 10, 1],
                  tips=False,
                  axis_config={"stroke_width": 5,
                               "include_ticks": True,}).scale(0.6).shift(1*DOWN+RIGHT)
        x_label = ax.get_x_axis_label(Tex(r"Time (t) $\rightarrow$"), edge=DOWN, direction=DOWN, buff=0.5).shift(LEFT)
        y_label = ax.get_y_axis_label(Tex(r"$\leftarrow$ Depth (y)"), edge=LEFT, direction=LEFT, buff=0.5).rotate(90*DEGREES)
        self.play(FadeIn(y_label, x_label))

        dots_0_coarse = VGroup()
        ny = 5
        dy = 10/(ny-1)
        for i in range(ny):
            dots_0_coarse.add(Circle(radius=0.05, color=WHITE, fill_opacity=1).move_to(ax.c2p(0,i*dy)))

        # Add time points in corase grid
        dots_coarse = VGroup(dots_0_coarse)
        for j in range(2, 10, 2):
            dots_j_coarse = VGroup()
            ny = 5
            dy = 10/(ny-1)
            # color_fn = color_gradient(["#b40426", WHITE, "#3b4cc0"], ny)
            for i in range(ny):
                dots_j_coarse.add(Circle(radius=0.05, color=WHITE, fill_opacity=1).move_to(ax.c2p(j*0.1,i*dy)))
            dots_coarse.add(dots_j_coarse)
        self.play(FadeIn(dots_coarse))
        self.wait(1)

        dy_arrow = DoubleArrow(start=ax.c2p(0,(ny-1)*dy + 0.5), end=ax.c2p(0,(ny-2)*dy - 0.5), color=MY_GOLD)
        dy_text = Tex(r"dy", color=MY_GOLD).scale(0.75).next_to(dy_arrow, RIGHT)
        dt_arrow = DoubleArrow(start=ax.c2p(0,(ny-1)*dy + 0.5), end=ax.c2p(1/4,(ny-1)*dy + 0.5), color=MY_GOLD).shift(0.25*LEFT)
        dt_text = Tex(r"dt", color=MY_GOLD).scale(0.75).next_to(dt_arrow, UP)
        self.play(Create(dt_arrow), Write(dt_text), Create(dy_arrow), Write(dy_text))
        self.wait(1)

        comment = Tex(r"Grid Velocity= $\frac{dy}{dt}$").next_to(title, DOWN)
        self.play(Write(comment))
        self.wait(1)

        cfl = Tex(r"$c < \frac{dy}{dt}$").scale(1.25).next_to(title, DOWN)
        self.play(ReplacementTransform(comment, cfl))
        self.wait(1)

        objects_to_keep = [title, cfl]  # Replace with your actual objects
        self.play(
            *[FadeOut(mob) for mob in self.mobjects if mob not in objects_to_keep], 
        )
        self.wait(2)

        # FadeOut
        self.play(
            *[FadeOut(mob)for mob in self.mobjects]
        )

        self.wait(2)