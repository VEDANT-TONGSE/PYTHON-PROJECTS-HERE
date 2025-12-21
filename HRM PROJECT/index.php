<?php
session_start();

?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
    <!-- Meta, title, CSS, favicons, etc. -->
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1">

    <title>Human Resource Management Sysytem</title>

    <!-- Bootstrap -->
    <link href="resource/css/bootstrap.css" rel="stylesheet">
    <!-- Font Awesome -->
    <link href="resource/css/font-awesome.css" rel="stylesheet">
    <!-- NProgress -->
    <link href="resource/css/nprogress.css" rel="stylesheet">
    <!-- Animate.css -->
    <link href="resource/css/animate.css" rel="stylesheet">

    <!-- Custom Theme Style -->
    <link href="resource/css/custom.css" rel="stylesheet">
</head>

<body class="login">
<div>
    <a class="hiddenanchor" id="signup"></a>
    <a class="hiddenanchor" id="signin"></a>

    <div class="login_wrapper">
        <div class="animate form login_form">
            <section class="login_content">
                <form action="src/store/Login.php" method="post">
                    <h1>Log in your Account</h1>
                    <h5>
                        <?php
                        if(!empty($_GET['msg'])) {
                            $var=$_GET['msg'];
                            echo "<div class=\"alert alert-danger fade in alert-dismissable\">
                                    <a href=\"#\" class=\"close\" data-dismiss=\"alert\" aria-label=\"close\" title=\"close\">×</a>
                                    $var
                                    </div>";
                        }
                        ?>
                    </h5>
                    <div>
                        <input type="text" id="userName" name="userName" class="form-control" placeholder="Username" required="" />
                    </div>
                    <div>
                        <input type="password" id="password" name="password" class="form-control" placeholder="Password" required="" />
                    </div>
                    <div>
                        <button type="submit" name="login" class="btn btn-default submit">Log in</button>
                        <a class="reset_pass" href="#signup">Lost your password ?</a>
                    </div>

                    <div class="clearfix"></div>

                    <div class="separator">
<!--                        <p class="change_link">New to site?-->
<!--                            <a href="#signup" class="to_register"> Create Account </a>-->
<!--                        </p>-->
<!---->
<!--                        <div class="clearfix"></div>-->
<!--                        <br />-->

                        <div>
                            <h1><i class="fa fa-paw"></i> SSVL!</h1>
                            <p>©2025 All Rights Reserved. Sandwitch Group</p>
                        </div>
                    </div>
                </form>
            </section>
        </div>

        <div id="register" class="animate form registration_form">
            <section class="login_content">
                <form>
                    <h1>Recovery Password</h1>
                    <div>
                        <input type="text" class="form-control" placeholder="Username" required="" />
                    </div>
                    <div>
                        <input type="email" class="form-control" placeholder="Email" required="" />
                    </div>
                    <div>
                        <input type="password" class="form-control" placeholder="Password" required="" />
                    </div>
                    <div>
                        <a class="btn btn-default submit" href="index.html">Submit</a>
                    </div>

                    <div class="clearfix"></div>

                    <div class="separator">
                        <p class="change_link">Already a member ?
                            <a href="#signin" class="to_register"> Log in </a>
                        </p>

                        <div class="clearfix"></div>
                        <br />

                        <div>
                            <h1><i class="fa fa-paw"></i> SSVL !</h1>
                            <p>©2025 All Rights Reserved. Sandwitch Group</p>
                        </div>
                    </div>
                </form>
            </section>
        </div>
    </div>
</div>
</body>
</html>

