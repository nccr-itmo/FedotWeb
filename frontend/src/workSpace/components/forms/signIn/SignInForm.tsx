import React, { memo } from "react";
import scss from "./signInForm.module.scss";
import { useFormik } from "formik";
import { useDispatch, useSelector } from "react-redux";
import clsx from "clsx";

import { Grid } from "@material-ui/core";

import { validationSchema } from "./validationSiginIn";
import TextFieldSignInFormik from "../../../../ui/formik/textFields/TextFieldSignInFormik";
import ButtonSiginInUpFormik from "../../../../ui/formik/buttons/ButtonSiginInUpFormik";
import { authUser } from "../../../../redux/reducers/auth/authReducer";
import { StateType } from "../../../../redux/store";

const SignInForm = () => {
  const dispatch = useDispatch();

  const { user_login_error } = useSelector((state: StateType) => state.auth);

  const formik = useFormik({
    initialValues: {
      login: "",
      password: "",
    },
    validationSchema: validationSchema,
    onSubmit: (values) => {
      dispatch(authUser(values.login, values.password));
    },
  });

  return (
    <form onSubmit={formik.handleSubmit}>
      <Grid container spacing={0}>
        <Grid item xs={12} className={scss.textFieldPosition}>
          <TextFieldSignInFormik
            name="login"
            value={formik.values.login}
            type="text"
            label="Login"
            error={(formik.errors.login && formik.touched.login) as boolean}
            errorText={
              ((formik.errors.login && formik.touched.login) as boolean)
                ? (formik.errors.login as string)
                : ""
            }
            onChange={formik.handleChange}
          />
        </Grid>
        <Grid item xs={12} className={scss.textFieldPosition}>
          <TextFieldSignInFormik
            name="password"
            value={formik.values.password}
            type="password"
            label="Password"
            error={
              (formik.errors.password && formik.touched.password) as boolean
            }
            errorText={
              ((formik.errors.password && formik.touched.password) as boolean)
                ? (formik.errors.password as string)
                : ""
            }
            onChange={formik.handleChange}
          />
        </Grid>
        <Grid item xs={12} className={scss.errorPosition}>
          <p
            className={
              user_login_error ? scss.error : clsx(scss.error, scss.errorNone)
            }
          >
            Authorization error!
          </p>
        </Grid>
        <Grid item xs={12} className={scss.buttonPosition}>
          <ButtonSiginInUpFormik
            disabled={
              Boolean(formik.errors.login) || Boolean(formik.errors.password)
            }
          >
            Sign In
          </ButtonSiginInUpFormik>
        </Grid>
      </Grid>
    </form>
  );
};

export default memo(SignInForm);
