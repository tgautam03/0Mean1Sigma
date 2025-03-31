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

class Domain3Dto1D(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=75*DEGREES, theta=-55*DEGREES)
        
        # Create 3D axes
        cube = Cube(side_length=3, fill_opacity=0, stroke_color=MY_GOLD, stroke_width=2)

        self.play(Create(cube))
        self.wait(1)

        # Create a solid plane along the x-z dimensions
        plane = Surface(
            lambda u, v: np.array([0, u, v]),
            u_range=[-1.5, 1.5],
            v_range=[-1.5, 1.5],
            resolution=(21, 21),
            fill_opacity=0.5,
            stroke_width=0
        ).set_color(WHITE)

        self.play(Create(plane))
        self.wait(1)

        # Create a vertical line through the center of the plane (z-direction)
        center_line = Line3D(
            start=np.array([0, 0, -1.5]),  # Start at the bottom of the plane in z
            end=np.array([0, 0, 1.5]),    # End at the top of the plane in z
            color=WHITE,
            thickness=0.02
        )

        self.play(FadeIn(center_line))

        self.wait(1)

        # Animate camera to front view
        self.move_camera(
            phi=90*DEGREES,      # Remove elevation angle
            theta=0*DEGREES,  # Face-on view
            frame_center=ORIGIN,
            zoom=2,
            run_time=1.5
        )
        self.wait(1)

        # FadeOut
        self.play(
            *[FadeOut(mob)for mob in self.mobjects]
        )

class Discretization(Scene):
    def construct(self):
        # Opening
        opening = Tex(r"Chapter 2: Building the Simulator", color=MY_GOLD).scale(1.5)
        self.play(FadeIn(opening))
        self.wait(1)
        self.play(FadeOut(opening))
        self.wait(2)

        wave_eq = Tex(r"$\frac{\partial^2 u(y,t)}{\partial t^2}$", r"$=$", r"$c^2(y)$", 
                      r"$\bigg[\frac{\partial^2 u(y,t)}{\partial y^2}\bigg]$", r"$+$", r"$s(y,t)$", color=WHITE)
        self.play(Write(wave_eq))
        self.play(wave_eq.animate.to_edge(UP).shift(0.25*DOWN))
        self.wait(1)

        ###########################################################################################
        ########################################### Grid ##########################################
        ###########################################################################################
        ax = Axes(x_range=[0, 1, 0.1],
                  y_range=[0, 10, 1],
                  tips=False,
                  axis_config={"stroke_width": 5,
                               "include_ticks": True,}).scale(0.6).shift(1*DOWN)
        x_label = ax.get_x_axis_label(Tex(r"Time (t) $\rightarrow$"), edge=DOWN, direction=DOWN, buff=0.5).shift(LEFT)
        y_label = ax.get_y_axis_label(Tex(r"$\leftarrow$ Depth (y)"), edge=LEFT, direction=LEFT, buff=0.5).rotate(90*DEGREES)

        # Change from finest to coarse grid
        dots_0_fine = VGroup()
        ny = 101
        dy = 10/(ny-1)
        # color_fn = color_gradient(["#b40426", WHITE, "#3b4cc0"], ny)
        for i in range(ny):
            dots_0_fine.add(Circle(radius=0.05, color=WHITE, fill_opacity=1).move_to(ax.c2p(0,i*dy)))
        self.play(Create(dots_0_fine), FadeIn(y_label), wave_eq[2].animate.set_color(MY_GOLD))

        dots_0_coarse = VGroup()
        ny = 5
        dy = 10/(ny-1)
        # color_fn = color_gradient(["#b40426", WHITE, "#3b4cc0"], ny)
        for i in range(ny):
            dots_0_coarse.add(Circle(radius=0.05, color=WHITE, fill_opacity=1).move_to(ax.c2p(0,i*dy)))

        self.play(FadeOut(dots_0_fine), FadeIn(dots_0_coarse))
        self.wait(1)

        dy_arrow = DoubleArrow(start=ax.c2p(0,(ny-1)*dy + 0.5), end=ax.c2p(0,(ny-2)*dy - 0.5), color=MY_GOLD)
        dy_text = Tex(r"dy").scale(0.75).next_to(dy_arrow, RIGHT)
        self.play(Create(dy_arrow), Write(dy_text))
        self.wait(1)

        self.play(Create(x_label), FadeOut(dy_arrow, dy_text))

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
        self.play(FadeIn(dots_coarse[1:]))
        self.wait(1)

        # Full fine grid
        dots_fine = VGroup()
        nt = 41
        ny = 21
        for j in range(nt):
            dots_j_fine = VGroup()
            dy = 10/(ny-1)
            dt = 1/(nt-1)
            # color_fn = color_gradient(["#b40426", WHITE, "#3b4cc0"], ny)
            for i in range(ny):
                dots_j_fine.add(Circle(radius=0.05, color=WHITE, fill_opacity=1).move_to(ax.c2p(j*dt,i*dy)))
            dots_fine.add(dots_j_fine)

        self.play(ReplacementTransform(dots_coarse, dots_fine), wave_eq[2].animate.set_color(WHITE))
        self.wait(1)

        # Where src?
        src_loc_arrow = Arrow(start=RIGHT, end=LEFT, color=MY_GOLD).next_to(dots_fine[-1][-1])
        src_loc_text = Tex(r"s(y,t)", color=MY_GOLD).scale(1).next_to(src_loc_arrow, RIGHT)
        self.play(Create(src_loc_arrow), Write(src_loc_text), wave_eq[-1].animate.set_color(MY_GOLD))
        self.play(VGroup(src_loc_arrow, src_loc_text).animate.next_to(dots_fine[-1][0]), run_time=1)
        self.play(VGroup(src_loc_arrow, src_loc_text).animate.next_to(dots_fine[-1][16]), run_time=1)
        self.play(VGroup(src_loc_arrow, src_loc_text).animate.next_to(dots_fine[-1][10]), run_time=1)
        self.wait(1)
        
        fade_out_dots = []
        for j in range(nt):
            for i in range(ny):
                if i != 10:
                    fade_out_dots.append(FadeOut(dots_fine[j][i]))
        self.play(*fade_out_dots, x_label.animate.next_to(dots_fine[int(nt/2)][int(ny/2)], DOWN))
        self.wait(1)

        # List of objects you want to keep
        objects_to_keep = [wave_eq]  # Replace with your actual objects
        self.play(
            *[FadeOut(mob) for mob in self.mobjects if mob not in objects_to_keep], 
        )
        self.wait(2)

        self.play(wave_eq[-1].animate.set_color(WHITE))
        self.play(wave_eq.animate.move_to(ORIGIN))

        # Transition
        dt_box = SurroundingRectangle(wave_eq[0], color=MY_GOLD)
        dx_box = SurroundingRectangle(wave_eq[3], color=MY_GOLD)
        self.play(Create(VGroup(dt_box, dx_box)))
        self.wait(1)

        title = Title("Taylor Approximation")
        title[0].move_to(ORIGIN)
        self.play(ReplacementTransform(VGroup(wave_eq, dt_box, dx_box), title[0]))
        self.wait(1)
        self.play(title[0].animate.to_edge(UP))
        self.play(Write(title[1]))
        self.wait(1)

        return super().construct()

class Taylor_Series(Scene):
    def construct(self):
        # Title
        title = Title("Taylor Approximation")
        title[0].move_to(ORIGIN)
        title[0].to_edge(UP)
        self.add(title)
        self.wait(1)
        
        # Taylor series
        f = Tex(r"$f(x)=\sin(x) + e^x$").next_to(title, DOWN).shift(0.25*DOWN).to_edge(LEFT)
        self.play(Write(f))
        self.wait(1)
        func_f = lambda x: np.sin(x) + np.exp(x)
        f_1_exp = Tex(r"$f(1)=$")
        f_1_val = Tex(str(func_f(1))).next_to(f_1_exp, RIGHT)
        f_1 = VGroup(f_1_exp, f_1_val).next_to(f, RIGHT).to_edge(RIGHT)
        self.play(Write(f_1))
        self.wait(1)
        f_taylor = Tex(r"$f(1+dx)=f(1)$",r"$+f'(1) \frac{dx}{1!}$",r"$+f''(1) \frac{dx^2}{2!}$",r"$+f'''(1) \frac{dx^3}{3!}$",r"$+f''''(1) \frac{dx^4}{4!}$",r"$+f'''''(1) \frac{dx^5}{5!}$",r"$+\cdots$").scale(0.75).next_to(VGroup(f_1, f), DOWN)
        self.play(Write(f_taylor[0][:7]))
        self.wait(1)
        self.play(Write(f_taylor[0][7:]), Write(f_taylor[1:]))
        self.wait(1)
        self.play(f_taylor[2:].animate.set_opacity(0.25))
        self.wait(1)
        func_f_derivs = [lambda x: np.cos(x) + np.exp(x), 
                        lambda x: -1*np.sin(x) + np.exp(x),
                        lambda x: -1*np.cos(x) + np.exp(x),
                        lambda x: np.sin(x) + np.exp(x),
                        lambda x: np.cos(x) + np.exp(x)]
        def func_f_1dx(dx, n):
            out = func_f(1)
            for i in range(1, n+1):
                out += func_f_derivs[i-1](1)*(dx**i)/math.factorial(i)
            return out
        
        dx_ = ValueTracker(1)
        dx_exp = Tex(r"$dx=$")
        dx_val = always_redraw(lambda: Tex(str(dx_.get_value())).next_to(dx_exp, RIGHT)) 
        dx = VGroup(dx_exp, dx_val).next_to(f_taylor, DOWN).shift(LEFT)
        self.play(Write(dx))
        self.wait(1)

        f_1dx_exp_ = Tex("True Value: ", r"$f(1+dx)=$")
        f_1dx_val_ = always_redraw(lambda: Tex(str(func_f(1+dx_.get_value()))).next_to(f_1dx_exp_, RIGHT))
        f_1dx_ = VGroup(f_1dx_exp_, f_1dx_val_).next_to(dx, DOWN).shift(DOWN)
        f_1dx_exp_.color = YELLOW
        self.play(Write(f_1dx_))
        
        n = ValueTracker(1)
        f_1dx_exp = Tex("Taylor Series Approximation: ", r"$f(1+dx)=$")
        f_1dx_val = always_redraw(lambda: Tex(str(round(func_f_1dx(dx_.get_value(), int(n.get_value())), 15))).next_to(f_1dx_exp, RIGHT))
        f_1dx = VGroup(f_1dx_exp, f_1dx_val).next_to(f_1dx_, DOWN).to_edge(LEFT)
        f_1dx_exp.color = RED_A

        diff_exp = Tex("Difference b/w True Value and Approximation: ").scale(0.85)
        diff_val = always_redraw(lambda: Tex(str(round(func_f(1+dx_.get_value()) - func_f_1dx(dx_.get_value(), int(n.get_value())), 15))).next_to(diff_exp, RIGHT))
        diff = VGroup(diff_exp, diff_val).next_to(f_1dx, DOWN).to_edge(LEFT)

        self.play(Write(f_1dx), Write(diff))
        self.wait(1)
        self.play(dx_.animate.set_value(0.5), run_time=2)
        self.wait(1)
        self.play(dx_.animate.set_value(0.1), run_time=2)
        self.wait(1)
        self.play(dx_.animate.set_value(0.05), run_time=2)
        self.wait(1)
        self.play(dx_.animate.set_value(0.01), run_time=2)
        self.wait(1)
        self.play(dx_.animate.set_value(0.005), run_time=2)
        self.wait(1)
        self.play(dx_.animate.set_value(0.001), run_time=2)
        self.wait(1)
        self.play(n.animate.set_value(2), f_taylor[2].animate.set_opacity(1))
        self.wait(1)
        self.play(n.animate.set_value(3), f_taylor[3].animate.set_opacity(1))
        self.wait(1)
        self.play(n.animate.set_value(4), f_taylor[4].animate.set_opacity(1))
        self.wait(1)
        self.play(n.animate.set_value(5), f_taylor[5].animate.set_opacity(1))
        self.wait(1)
        self.play(FadeOut(dx, f_1dx_, f_1dx, diff))
        self.wait(1)

        add = Tex(r"$+$").next_to(f_taylor, DOWN)
        f_taylor_ = Tex(r"$f(1-dx)=f(1)$",r"$-f'(1) \frac{dx}{1!}$",r"$+f''(1) \frac{dx^2}{2!}$",r"$-f'''(1) \frac{dx^3}{3!}$",r"$+f''''(1) \frac{dx^4}{4!}$",r"$-f'''''(1) \frac{dx^5}{5!}$",r"$+\cdots$").scale(0.75).next_to(add, DOWN)
        self.play(Write(f_taylor_))
        self.wait(1)
        self.play(Write(add))
        self.wait(1)
        self.play(f_taylor[1].animate.set_opacity(0.25), f_taylor_[1].animate.set_opacity(0.25), f_taylor[3].animate.set_opacity(0.25), f_taylor_[3].animate.set_opacity(0.25), f_taylor[5].animate.set_opacity(0.25), f_taylor_[5].animate.set_opacity(0.25))
        self.wait(1)
        f_2deriv = Tex(r"$f(1+dx)+f(1-dx)=2f(1)$",r"$+2f''(1) \frac{dx^2}{2!}$",r"$+2f''''(1) \frac{dx^4}{4!}$",r"$+\cdots$").scale(1).next_to(f_taylor_, DOWN).shift(DOWN)
        self.play(Write(f_2deriv))
        self.wait(1)
        f_2deriv_ = Tex(r"$\frac{f(1+dx)-2f(1)+f(1-dx)}{dx^2}=$",r"$f''(1)$",r"$+2f''''(1) \frac{dx^2}{4!}$",r"$+\cdots$").scale(1).next_to(f_taylor_, DOWN).shift(DOWN)
        self.play(ReplacementTransform(f_2deriv, f_2deriv_))
        self.wait(1)
        self.play(f_2deriv_[-2:].animate.set_opacity(0.25))
        self.wait(1)
        f_2deriv = Tex(r"$\frac{f(1+dx)-2f(1)+f(1-dx)}{dx^2}=$",r"$f''(1)$",r"$+O(dx^2)$").scale(1).next_to(f_taylor_, DOWN).shift(DOWN)
        self.play(TransformMatchingTex(f_2deriv_, f_2deriv))
        self.wait(1)
        self.play(
            *[FadeOut(mob)for mob in self.mobjects]
        )
        self.wait(1)

        return super().construct()


class Difference_Eq(Scene):
    def construct(self):
        # Deriv approx
        f_2deriv_1 = Tex(r"$f''(x) \approx \frac{f(x+dx)-2f(x)+f(x-dx)}{dx^2}$", opacity=0.5).scale(1).to_edge(UP).shift(0.25*DOWN)
        f_2deriv_2 = Tex(r"$f''(x) \approx \frac{f(x+dx)-2f(x)+f(x-dx)}{dx^2}$", opacity=0.5).scale(1).to_edge(UP).shift(0.25*DOWN)

        self.play(FadeIn(VGroup(f_2deriv_1, f_2deriv_2)))

        #############################################################################
        ################################## Wave Equation ############################
        #############################################################################
        wave_eq = Tex(r"$\frac{\partial^2 u(y,t)}{\partial t^2}$", r"$=$", r"$c(y)^2$", 
                      r"$\bigg[\frac{\partial^2 u(y,t)}{\partial y^2}\bigg]$", r"$+$", r"$s(y,t)$", color=WHITE).scale(0.9)
        self.play(Write(wave_eq))

        wave_eq_discrete_1 = Tex(r"$\frac{u(y,t+dt)-2u(y,t)+u(y,t-dt)}{dt^2}$", r"$=$", r"$c(y)^2$", 
                                 r"$\bigg[\frac{\ u(y,t)}{\partial y^2}\bigg]$", r"$+$", r"$s(y,t)$", color=WHITE).scale(0.9)
        self.play(FadeOut(f_2deriv_1, target_position=wave_eq[0]), ReplacementTransform(wave_eq, wave_eq_discrete_1))
        self.wait(1)

        wave_eq_discrete_2 = Tex(r"$\frac{u(y,t+dt)-2u(y,t)+u(y,t-dt)}{dt^2}$", r"$=$", r"$c(y)^2$", 
                                 r"$\bigg[\frac{u(y+dy,t)-2u(y,t)+u(y-dy,t)}{dy^2}\bigg]$", r"$+$", r"$s(y,t)$", color=WHITE).scale(0.9)
        self.play(FadeOut(f_2deriv_2, target_position=wave_eq_discrete_1[3]), ReplacementTransform(wave_eq_discrete_1, wave_eq_discrete_2))
        self.wait(1)

        # Past, present, future
        future_terms = wave_eq_discrete_2[0][:9]
        future_text = Tex("Future", color=GREEN).scale(0.7).next_to(future_terms, UP)
        self.play(future_terms.animate.set_color(GREEN), Write(future_text))

        present_terms = VGroup(wave_eq_discrete_2[0][11:17], wave_eq_discrete_2[3][1:10], wave_eq_discrete_2[3][12:18], 
                               wave_eq_discrete_2[3][20:29])
        present_text = Tex("Present", color=BLUE).scale(0.7).next_to(present_terms, UP)
        self.play(present_terms.animate.set_color(BLUE), Write(present_text))

        past_terms = wave_eq_discrete_2[0][18:27]
        past_text = Tex("Past", color=RED).scale(0.7).next_to(past_terms, UP)
        self.play(past_terms.animate.set_color(RED), Write(past_text))
        self.wait(1)

        # Explicit solution
        wave_eq_discrete_exp = Tex(r"$u(y,t+dt)$", r"$=$", r"$2u(y,t)$", r"$-u(y,t-dt)$", r"$+c(y)^2 dt^2$", 
                                   r"$\bigg[\frac{u(y+dy,t)-2u(y,t)+u(y-dy,t)}{dy^2}\bigg]$", r"$+$", r"$dt^2 s(y,t)$", color=WHITE).scale(0.8)
        wave_eq_discrete_exp[0].set_color(GREEN)
        wave_eq_discrete_exp[2].set_color(BLUE)
        wave_eq_discrete_exp[5][1:10].set_color(BLUE)
        wave_eq_discrete_exp[5][12:18].set_color(BLUE)
        wave_eq_discrete_exp[5][20:29].set_color(BLUE)
        wave_eq_discrete_exp[3][1:].set_color(RED)

        self.play(ReplacementTransform(wave_eq_discrete_2, wave_eq_discrete_exp), 
                  future_text.animate.next_to(wave_eq_discrete_exp[0], UP),
                  present_text.animate.next_to(VGroup(wave_eq_discrete_exp[2], wave_eq_discrete_exp[5]), UP),
                  past_text.animate.next_to(wave_eq_discrete_exp[3][1:], UP))
        self.wait(1)
        self.play(VGroup(wave_eq_discrete_exp, future_text, present_text, past_text).animate.to_edge(UP).shift(0.25*DOWN))
        self.wait(1)

        # Full fine grid
        ax = Axes(x_range=[0, 1, 0.1],
                  y_range=[0, 10, 1],
                  tips=False,
                  axis_config={"stroke_width": 5,
                               "include_ticks": True,}).scale(0.4).to_edge(DOWN).to_edge(RIGHT).shift(UP)
        x_label = ax.get_x_axis_label(Tex(r"Time (t) $\rightarrow$"), edge=DOWN, direction=DOWN, buff=0.1).scale(0.7)
        y_label = ax.get_y_axis_label(Tex(r"$\leftarrow$ Depth (y)"), edge=LEFT, direction=LEFT, buff=0.1).scale(0.7).rotate(90*DEGREES).shift(RIGHT)

        dots_fine = VGroup()
        nt = 11
        ny = 5
        for j in range(nt):
            dots_j_fine = VGroup()
            dy = 10/(ny-1)
            dt = 1/(nt-1)
            # color_fn = color_gradient(["#b40426", WHITE, "#3b4cc0"], ny)
            for i in range(ny):
                dots_j_fine.add(Circle(radius=0.05, color=WHITE, fill_opacity=1).move_to(ax.c2p(j*dt,i*dy)))
            dots_fine.add(dots_j_fine)

        self.play(Create(dots_fine), Write(x_label), Write(y_label))
        self.wait(1)

        comment_1 = Tex("2D Grid", color=MY_GOLD).next_to(dots_fine, UP)
        self.play(Write(comment_1))
        self.wait(1)

        circ = Circle(radius=0.1, color=MY_GOLD).move_to(dots_fine[0][0])
        index_2d = Tex("u(4,0)", color=MY_GOLD).scale(0.7).next_to(circ, DOWN)
        self.play(Create(circ), Write(index_2d))
        self.wait(1)

        # MatrixFP32
        self.play(VGroup(dots_fine, x_label, y_label).animate.set_opacity(0), FadeOut(comment_1, circ, index_2d))
        self.wait(1)

        # Solve Sim
        temp_box = SurroundingRectangle(wave_eq_discrete_exp[5], color=MY_GOLD)
        self.play(Create(temp_box))
        self.wait(1)
        temp_box_2 = SurroundingRectangle(wave_eq_discrete_exp[7], color=MY_GOLD)
        self.play( ReplacementTransform(temp_box, temp_box_2))
        self.wait(1)
        self.play(FadeOut(temp_box_2))
        self.wait(1)

        self.play(VGroup(x_label, y_label).animate.set_opacity(1), dots_fine.animate.set_opacity(0.2))
        self.wait(1)

        initial_cond_box = SurroundingRectangle(dots_fine[:2], color=MY_GOLD, buff=0.2)
        initial_cond_text = Tex("Initial Values = 0", color=MY_GOLD).scale(0.5).next_to(initial_cond_box, UP)
        self.play(Create(initial_cond_box), Write(initial_cond_text), dots_fine[:2].animate.set_opacity(1))
        self.wait(1)
        self.play(FadeOut(initial_cond_box, initial_cond_text))
        self.wait(1)

        boundary_pts_1 = VGroup(dots_fine[0][0],
                                dots_fine[1][0],
                                dots_fine[2][0],
                                dots_fine[3][0],
                                dots_fine[4][0],
                                dots_fine[5][0],
                                dots_fine[6][0],
                                dots_fine[7][0],
                                dots_fine[8][0],
                                dots_fine[9][0],
                                dots_fine[10][0])
        boundary_pts_2 = VGroup(dots_fine[0][4],
                                dots_fine[1][4],
                                dots_fine[2][4],
                                dots_fine[3][4],
                                dots_fine[4][4],
                                dots_fine[5][4],
                                dots_fine[6][4],
                                dots_fine[7][4],
                                dots_fine[8][4],
                                dots_fine[9][4],
                                dots_fine[10][4])
        boundary_cond_box_1 = SurroundingRectangle(boundary_pts_1, color=MY_GOLD, buff=0.2)
        boundary_cond_box_2 = SurroundingRectangle(boundary_pts_2, color=MY_GOLD, buff=0.2)
        boundary_cond_text = Tex("Boundaries", color=MY_GOLD).scale(0.5).next_to(boundary_cond_box_2, UP)
        self.play(Create(boundary_cond_box_1), Create(boundary_cond_box_2), Write(boundary_cond_text))
        self.wait(1)
        self.play(boundary_pts_1.animate.set_opacity(1), boundary_pts_2.animate.set_opacity(1))
        self.wait(1)
        self.play(FadeOut(boundary_cond_box_1, boundary_cond_box_2, boundary_cond_text))
        self.wait(1)

        circ = Circle(radius=0.1, color=MY_GOLD).move_to(dots_fine[2][ny-2])
        for j in range(2, nt):
            for i in reversed(range(1, ny-1)):
                if j == 2 and i == ny-2:
                    self.play(Create(circ))
                    self.play(dots_fine[j][i].animate.set_opacity(1), run_time=0.1)
                else:
                    self.play(circ.animate.move_to(dots_fine[j][i]), run_time=0.1)
                    self.play(dots_fine[j][i].animate.set_opacity(1), run_time=0.1)
        
        self.play(FadeOut(dots_fine, x_label, y_label, circ))
        self.wait(1)

        self.play(
            *[FadeOut(mob)for mob in self.mobjects]
        )

        self.wait(1)
        return super().construct()