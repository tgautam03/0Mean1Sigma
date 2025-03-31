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

class WaveProp3d(ThreeDScene):
    def construct(self):
        MY_GOLD = "#DAA520"
        self.set_camera_orientation(phi=70*DEGREES, theta=-40*DEGREES)
        
        # Create 3D axes
        cube = Cube(side_length=4.75, fill_opacity=0, stroke_color=MY_GOLD, stroke_width=2)

        self.play(Create(cube))
        self.wait(1)

        # Create a solid plane along the x-z dimensions
        plane = Surface(
            lambda u, v: np.array([u, v, 4.75/2]),
            u_range=[-4.75/2, 4.75/2],
            v_range=[-4.75/2, 4.75/2],
            resolution=(21, 21),
            fill_opacity=0.5,
            stroke_width=0
        ).set_color(WHITE)

        self.play(Create(plane))
        self.wait(1)

        # Animate camera to front view
        self.move_camera(
            phi=90*DEGREES,      # Remove elevation angle
            theta=0*DEGREES,  # Face-on view
            frame_center=ORIGIN,
            zoom=1,
            run_time=1.5
        )
        self.wait(1)

        plane_2 = Surface(
            lambda u, v: np.array([0, u, v]),
            u_range=[-4.75/2, 4.75/2],
            v_range=[-4.75/2, 4.75/2],
            resolution=(21, 21),
            fill_opacity=0.5,
            stroke_width=0
        ).set_color(WHITE)

        self.play(Create(plane_2))
        self.wait(1)

        self.wait(5)

        # FadeOut
        self.play(
            *[FadeOut(mob)for mob in self.mobjects]
        )

class Motivation(Scene):
    def construct(self):
        # Show 3D wave sim
        self.wait(5)

        # List of content
        box_1 = Rectangle(height=2, width=3.5, color=MY_GOLD, stroke_width=10).shift(4.5*LEFT+2*UP)
        text_10 = Tex("Chapter 1", color=WHITE).scale(0.7).next_to(box_1, UP)
        text_11 = Tex("The Wave Equation", color=WHITE).scale(0.7).next_to(box_1, DOWN)
        self.play(Create(box_1))
        self.play(Write(text_10), Write(text_11))
        self.wait(1)

        box_2 = Rectangle(height=2, width=3.5, color=MY_GOLD, stroke_width=10).shift(4.5*RIGHT+2*UP)
        text_20 = Tex("Chapter 2", color=WHITE).scale(0.7).next_to(box_2, UP)
        text_21 = Tex("Building the Simulator", color=WHITE).scale(0.7).next_to(box_2, DOWN)
        self.play(Create(box_2))
        self.play(Write(text_20), Write(text_21))
        self.wait(1)

        box_3 = Rectangle(height=2, width=3.5, color=MY_GOLD, stroke_width=10)#.shift(4.5*RIGHT+2*UP)
        text_30 = Tex("Chapter 3", color=WHITE).scale(0.7).next_to(box_3, UP)
        text_31 = Tex(r"Realistic Simulation", color=WHITE).scale(0.7).next_to(box_3, DOWN)
        self.play(Create(box_3))
        self.play(Write(text_30), Write(text_31))
        self.wait(1)

        box_4 = Rectangle(height=2, width=3.5, color=MY_GOLD, stroke_width=10).shift(4.5*LEFT+2*DOWN)
        text_40 = Tex("Chapter 4", color=WHITE).scale(0.7).next_to(box_4, UP)
        text_41 = Tex("Improving Accuracy and Stability", color=WHITE).scale(0.65).next_to(box_4, DOWN)
        self.play(Create(box_4))
        self.play(Write(text_40), Write(text_41))
        self.wait(1)

        box_5 = Rectangle(height=2, width=3.5, color=MY_GOLD, stroke_width=10).shift(4.5*RIGHT+2*DOWN)
        text_50 = Tex("Chapter 5", color=WHITE).scale(0.7).next_to(box_5, UP)
        text_51 = Tex("3D Simulation", color=WHITE).scale(0.7).next_to(box_5, DOWN)
        self.play(Create(box_5))
        self.play(Write(text_50), Write(text_51))
        self.wait(1)

        self.wait(5)
        # FadeOut
        self.play(
            *[FadeOut(mob)for mob in self.mobjects]
        )