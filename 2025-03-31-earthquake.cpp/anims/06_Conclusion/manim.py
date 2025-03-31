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

class Conclusion(Scene):
    def construct(self):
        img_github = ImageMobject("GitHub.png").scale(0.75)
        box_github = SurroundingRectangle(img_github, color=MY_GOLD, stroke_width=10, buff=0.025)
        text_github = Tex("GitHub: tgautam03", color=MY_GOLD).next_to(box_github, DOWN)

        self.play(FadeIn(img_github), Create(box_github), Write(text_github), run_time=2)
        self.wait(2)

        self.play(FadeOut(img_github), FadeOut(box_github), FadeOut(text_github), run_time=2)
        
        img_blog = ImageMobject("Blog.png").scale(0.6)
        box_blog = SurroundingRectangle(img_blog, color=MY_GOLD, stroke_width=10, buff=0.025)
        text_blog = Tex("0mean1sigma.com", color=MY_GOLD).next_to(box_blog, DOWN)

        self.play(FadeIn(img_blog), Create(box_blog), Write(text_blog), run_time=2)
        self.wait(2)

        self.play(FadeOut(img_blog), FadeOut(box_blog), FadeOut(text_blog), run_time=2)

        dialogue = Tex("Let me know what you ", "think",".").scale(1.5)
        dialogue[1].set_color(RED)
        self.play(Write(dialogue))
        self.wait(2)
        self.play(
            *[FadeOut(mob)for mob in self.mobjects]
        )
        self.wait(1)

class Thumbnail(Scene):
    def construct(self):
        img_3d = ImageMobject("3d_1.png").scale(5).to_edge(DOWN)
        box_3d = SurroundingRectangle(img_3d, color=MY_GOLD, stroke_width=10)

        seis_img = ImageMobject("seismogram.png").scale(1.25).to_edge(UP).to_edge(LEFT)
        text_3d = Tex("Earthquake", color=WHITE).scale(1.75).next_to(seis_img, RIGHT).shift(0.3*DOWN)

        comment = Tex("3D SIMULATION", color=MY_GOLD).scale(2).move_to(box_3d).shift(2.5*LEFT+1.25*UP)

        self.add(img_3d, text_3d, seis_img, box_3d, comment)
        return super().construct()