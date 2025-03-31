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

class WaveEquation(Scene):
    def construct(self):
        # Opening
        opening = Tex(r"Chapter 1: The Wave Equation", color=MY_GOLD).scale(1.5)
        self.play(FadeIn(opening))
        self.wait(1)
        self.play(FadeOut(opening))

        #########################################################################################
        ######################################## Wave equation ##################################
        #########################################################################################
        wave_eq = Tex(r"$\frac{\partial^2 u(x,y,z,t)}{\partial t^2}$", r"$=$", r"$c^2(x,y,z)$", 
                      r"$\bigg[\frac{\partial^2 u(x,y,z,t)}{\partial x^2}+\frac{\partial^2 u(x,y,z,t)}{\partial y^2} + \frac{\partial^2 u(x,y,z,t)}{\partial z^2}\bigg]$", 
                      r"$+$", r"$s(x,y,z,t)$", color=WHITE).scale(0.85)
        self.play(Write(wave_eq))
        self.play(wave_eq.animate.to_edge(UP).shift(0.25*DOWN))

        ##########################################################################################
        ############################################# Source #####################################
        ##########################################################################################
        self.wait(1)
        self.play(wave_eq[-1].animate.set_color(MY_GOLD))
        self.wait(2)

        # src equation
        src_eq = Tex(r"$s(t)$", r"$=$", r"$-8 \cdot (t - t_0) \cdot f_0 \cdot e^{-(4 \cdot f_0 \cdot (t - t_0))^2}$", 
                     color=WHITE).to_edge(DOWN).shift(0.25*UP)
        self.play(Write(src_eq))
        self.wait(1)

        # f0 and t0
        self.play(src_eq[2][6:8].animate.set_color(MY_GOLD), 
                  src_eq[2][10:12].animate.set_color(MY_GOLD),
                  src_eq[2][18:20].animate.set_color(MY_GOLD))
        self.wait(1)

        ax = Axes(x_range=[0, 1, 0.02],
                  y_range=[-1, 1, 0.02],
                  tips=False,
                  axis_config={"stroke_width": 5,
                               "include_ticks": False,}).scale(0.6).next_to(src_eq, UP)
        x_label = ax.get_x_axis_label(Tex(r"Time $\rightarrow$"), edge=RIGHT, direction=DOWN, buff=0.5)
        y_label = ax.get_y_axis_label(Tex("Earthquake"), edge=LEFT, direction=LEFT, buff=0.5)
        
        # Let's see how f0 and t0 affects src
        t0 = ValueTracker(0.1)
        f0 = ValueTracker(20)
        def src_fn(t):
            return -8 * (t - t0.get_value()) * f0.get_value() * np.exp(-1 * 4*f0.get_value()*(t - t0.get_value()) * 4*f0.get_value()*(t - t0.get_value()))
        
        plot = always_redraw(lambda: ax.plot(src_fn, color="#DAA520", stroke_width=10)) 

        f0_text = Tex(r"$f_0$=").next_to(x_label, UP).shift(0.5*UP+LEFT)
        f0_show = always_redraw(lambda: DecimalNumber(f0.get_value(), num_decimal_places=2).next_to(f0_text, RIGHT))
        t0_text = Tex(r"$t_0$=").next_to(f0_text, UP)
        t0_show = always_redraw(lambda: DecimalNumber(t0.get_value(), num_decimal_places=2).next_to(t0_text, RIGHT))

        self.play(Create(ax), Create(x_label), Create(y_label))
        self.play(Write(plot), Write(f0_text), Write(t0_text), Write(f0_show), Write(t0_show))
        self.wait(1)

        # Animate changes in t0 and f0
        self.play(t0.animate.set_value(0.5), run_time=3)  # Shift t0
        self.play(f0.animate.set_value(5), run_time=3)  # Decrease f0
        self.wait(1)
        
        self.play(FadeOut(ax, x_label, y_label, plot, f0_text, f0_show, t0_text, t0_show, src_eq), 
                  wave_eq[-1].animate.set_color(WHITE))
        self.wait(1)

        ##########################################################################################
        ############################################# c(y) #######################################
        ##########################################################################################
        self.wait(2)
        self.play(wave_eq[2].animate.set_color(MY_GOLD))
        self.wait(2)

        comment = Tex(r"Different $c(x,y,z)$", color=MY_GOLD).scale(1).next_to(wave_eq, DOWN).shift(0.25*DOWN)
        window = Rectangle(height=3.5, width=7, color=MY_GOLD, stroke_width=10).next_to(comment, DOWN)
        credits = Tex("Credit: GemPy").next_to(window, DOWN)
        self.play(Write(comment), Create(window), Write(credits))
        self.wait(2)
        self.play(FadeOut(comment, window, credits))
        self.wait(1)

        ##########################################################################################
        ########################################## du(y,t) #######################################
        ##########################################################################################
        self.play(VGroup(wave_eq[0], wave_eq[3]).animate.set_color(MY_GOLD), wave_eq[2].animate.set_color(WHITE))
        self.wait(2)

        comment = Tex("Solve u(x,y,z,t)!", color=MY_GOLD)
        self.play(Write(comment))
        self.wait(2)
        
        # FadeOut
        self.play(
            *[FadeOut(mob)for mob in self.mobjects]
        )

        return super().construct()

# class WaveEquation(Scene):
#     def construct(self):
#         # Opening
#         opening = Tex(r"Chapter 1: The Wave Equation", color=MY_GOLD).scale(1.5)
#         self.play(FadeIn(opening))
#         self.wait(1)
#         self.play(FadeOut(opening))

#         # 3D to 1D
#         self.wait(2)

#         #########################################################################################
#         ######################################## Wave equation ##################################
#         #########################################################################################
#         wave_eq = Tex(r"$\frac{\partial^2 u(y,t)}{\partial t^2}$", r"$=$", r"$c^2(y)$", r"$\bigg[\frac{\partial^2 u(y,t)}{\partial y^2}\bigg]$", r"$+$", r"$s(y,t)$", color=WHITE).scale(1.35)
#         self.play(Write(wave_eq))
#         self.play(wave_eq.animate.to_edge(UP).shift(0.25*DOWN))

#         ##########################################################################################
#         ############################################# Source #####################################
#         ##########################################################################################
#         self.wait(1)
#         self.play(wave_eq[-1].animate.set_color(MY_GOLD))
#         self.wait(1)

#         # src equation
#         src_eq = Tex(r"$s(t)$", r"$=$", r"$-8 \cdot (t - t_0) \cdot f_0 \cdot e^{-(4 \cdot f_0 \cdot (t - t_0))^2}$", color=WHITE).to_edge(DOWN).shift(0.25*UP)
#         self.play(Write(src_eq))
#         self.wait(1)

#         # src code
#         src_code = Code(code_string='''src = -8 * (t - t0) * f0 * exp(-1 * 4*f0*(t - t0) * 4*f0*(t - t0))''',
#                         language="C++", add_line_numbers=False,
#                         formatter_style="dracula",
#                         paragraph_config={"font": "Monospace"}).scale(0.85)
#         self.play(Write(src_code[1]))
#         self.wait(1)
#         self.play(FadeOut(src_code[1]))
#         self.wait(1)

#         # f0 and t0
#         self.play(src_eq[2][6:8].animate.set_color(MY_GOLD), 
#                   src_eq[2][10:12].animate.set_color(MY_GOLD),
#                   src_eq[2][18:20].animate.set_color(MY_GOLD))
#         self.wait(1)

#         ax = Axes(x_range=[0, 1, 0.02],
#                   y_range=[-1, 1, 0.02],
#                   tips=False,
#                   axis_config={"stroke_width": 5,
#                                "include_ticks": False,}).scale(0.6).next_to(src_eq, UP)
#         x_label = ax.get_x_axis_label(Tex(r"Time $\rightarrow$"), edge=RIGHT, direction=DOWN, buff=0.5)
#         y_label = ax.get_y_axis_label(Tex("Energy"), edge=LEFT, direction=LEFT, buff=0.5)
        
#         # Let's see how f0 and t0 affects src
#         t0 = ValueTracker(0.1)
#         f0 = ValueTracker(20)
#         def src_fn(t):
#             return -8 * (t - t0.get_value()) * f0.get_value() * np.exp(-1 * 4*f0.get_value()*(t - t0.get_value()) * 4*f0.get_value()*(t - t0.get_value()))
        
#         plot = always_redraw(lambda: ax.plot(src_fn, color="#DAA520", stroke_width=10)) 

#         f0_text = Tex(r"$f_0$=").next_to(x_label, UP).shift(0.5*UP+LEFT)
#         f0_show = always_redraw(lambda: DecimalNumber(f0.get_value(), num_decimal_places=2).next_to(f0_text, RIGHT))
#         t0_text = Tex(r"$t_0$=").next_to(f0_text, UP)
#         t0_show = always_redraw(lambda: DecimalNumber(t0.get_value(), num_decimal_places=2).next_to(t0_text, RIGHT))

#         self.play(Create(ax), Create(x_label), Create(y_label))
#         self.play(Write(plot), Write(f0_text), Write(t0_text), Write(f0_show), Write(t0_show))
#         self.wait(1)

#         # Animate changes in t0 and f0
#         self.play(t0.animate.set_value(0.5), run_time=3)  # Shift t0
#         self.play(f0.animate.set_value(10), run_time=3)  # Increase f0

#         self.play(f0.animate.set_value(5), t0.animate.set_value(0.15), run_time=3)  # Increase f0
#         self.wait(1)
        
#         self.play(FadeOut(ax, x_label, y_label, plot, f0_text, f0_show, t0_text, t0_show, src_eq), 
#                   wave_eq[-1].animate.set_color(WHITE))
#         self.wait(1)

#         ##########################################################################################
#         ############################################# c(y) #######################################
#         ##########################################################################################
#         self.play(wave_eq[2].animate.set_color(MY_GOLD))
#         # Show wave sims
#         window_1 = Rectangle(height=4, width=4, color=MY_GOLD, stroke_width=10).shift(3*LEFT).to_edge(DOWN)
#         c_1 = Tex(r"c=5 km/s", color=MY_GOLD).next_to(window_1, UP)
#         window_2 = Rectangle(height=4, width=4, color=MY_GOLD, stroke_width=10).shift(3*RIGHT).to_edge(DOWN)
#         c_2 = Tex(r"c=10 km/s", color=MY_GOLD).next_to(window_2, UP)
#         self.play(Write(window_1), Write(window_2))
#         self.play(Write(c_1), Write(c_2))
#         self.wait(2)
#         self.play(FadeOut(window_1, window_2, c_1, c_2))
#         window_3 = Rectangle(height=4, width=6, color=MY_GOLD, stroke_width=10).to_edge(DOWN)
#         c_3 = Tex(r"Realistic Model", color=MY_GOLD).next_to(window_3, UP)
#         self.play(Write(window_3), Write(c_3))
#         self.wait(2)

#         ##########################################################################################
#         ########################################## du(y,t) #######################################
#         ##########################################################################################
#         self.play(VGroup(wave_eq[0], wave_eq[3]).animate.set_color(MY_GOLD), wave_eq[2].animate.set_color(WHITE))
#         self.wait(2)
#         amp_arrow = Arrow(start=LEFT, end=2*RIGHT, color=MY_GOLD).next_to(window_3.get_edge_center(DOWN), UP).shift(LEFT)
#         amp_text = Tex("Displacement", color=MY_GOLD).scale(0.7).next_to(amp_arrow, RIGHT)
#         wave_arrow = Arrow(start=1.5*DOWN, end=1.5*UP, color=MY_GOLD).next_to(window_3.get_edge_center(LEFT), RIGHT).shift(0.25*RIGHT)
#         wave_text = Tex("Wave Propagation", color=MY_GOLD).scale(0.7).next_to(wave_arrow, LEFT).rotate(90 * DEGREES).shift(1.5*RIGHT)
#         self.play(Create(amp_arrow), Write(amp_text), Create(wave_arrow), Write(wave_text))
#         self.wait(2)
        
#         # FadeOut
#         self.play(
#             *[FadeOut(mob)for mob in self.mobjects]
#         )

#         return super().construct()