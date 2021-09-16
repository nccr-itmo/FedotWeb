import React, { memo, useCallback } from "react";
import { useDispatch, useSelector } from "react-redux";

import { makeStyles, createStyles } from "@material-ui/core/styles";
import { Fade } from "@material-ui/core";

import AddButton from "../../../../ui/buttons/AddButton";
import AddNodeForm from "../../../components/forms/addNode/AddNodeForm";
import { setEditOrAddFormOpened } from "../../../../redux/reducers/sandBox/sandBoxReducer";
import { StateType } from "../../../../redux/store";

const useStyles = makeStyles(() =>
  createStyles({
    root: {
      position: "relative",
    },
    button: {
      position: "absolute",
      top: -20,
      right: 0,
      zIndex: 2,
    },
  })
);

const AddNode = () => {
  const classes = useStyles();
  const dispatch = useDispatch();
  const { sandbox_edit_or_add_form_opened } = useSelector(
    (state: StateType) => state.sandBox
  );

  const onAddPointClick = useCallback(() => {
    if (sandbox_edit_or_add_form_opened.isOpen) {
      dispatch(setEditOrAddFormOpened({ isOpen: false, type: "new" }));
    } else {
      dispatch(setEditOrAddFormOpened({ isOpen: true, type: "new" }));
    }
  }, [dispatch, sandbox_edit_or_add_form_opened]);

  const setFromClosed = useCallback(() => {
    dispatch(setEditOrAddFormOpened({ isOpen: false, type: "new" }));
  }, [dispatch]);

  return (
    <div className={classes.root}>
      <div className={classes.button}>
        <AddButton onClick={onAddPointClick}>add Point</AddButton>
      </div>
      <Fade
        in={sandbox_edit_or_add_form_opened.isOpen}
        mountOnEnter
        unmountOnExit
      >
        <div>
          <AddNodeForm
            setFromClosed={setFromClosed}
            type={sandbox_edit_or_add_form_opened.type}
          />
        </div>
      </Fade>
    </div>
  );
};

export default memo(AddNode);
