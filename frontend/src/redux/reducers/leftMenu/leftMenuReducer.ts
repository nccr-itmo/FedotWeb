import { ThunkAction } from "redux-thunk";
import { StateType } from "../../store";

const PAGE_SELECTED = "PAGE_SELECTED";

let initialState = {
  page_selected: "Showcase" as "Showcase" | "Sandbox" | "FEDOT" | "Settings",
};

export type InitialStateType = typeof initialState;
type AllTyps = PageSelectedType;

const leftMenuReducer = (
  state = initialState,
  action: AllTyps
): InitialStateType => {
  switch (action.type) {
    case PAGE_SELECTED:
      return { ...state, page_selected: action.data };

    default:
      return state;
  }
};

type PageSelectedType = {
  type: typeof PAGE_SELECTED;
  data: "Showcase" | "Sandbox" | "FEDOT" | "Settings";
};

export const actionsLeftMenu = {
  pageSelected: (
    data: "Showcase" | "Sandbox" | "FEDOT" | "Settings"
  ): PageSelectedType => ({ type: PAGE_SELECTED, data }),
};

// type ThunkTypeAsync = ThunkAction<Promise<void>, StateType, unknown, AllTyps>;
type ThunkType = ThunkAction<void, StateType, unknown, AllTyps>;

export const selectPage = (
  page: "Showcase" | "Sandbox" | "FEDOT" | "Settings"
): ThunkType => {
  return (dispatch) => {
    dispatch(actionsLeftMenu.pageSelected(page));
  };
};

export default leftMenuReducer;
