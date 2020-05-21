from dataclasses import dataclass


@dataclass
class Coordinates:
    def __init__(self, column, row):
        self._column = column
        self._row = row

    @property
    def column(self) -> int:
        return self._column

    @property
    def row(self) -> int:
        return self._row

    @column.setter
    def column(self, i: int) -> None:
        self._column = i

    @row.setter
    def row(self, i: int) -> None:
        self._row = i


@dataclass
class Player:
    def __init__(self, head, shoulders, centroid, hand1, hand2, elbow1, elbow2, leg1, leg2, knee1, knee2):
        self._head = head
        self._shoulders = shoulders
        self._centroid = centroid
        self._hand1 = hand1
        self._hand2 = hand2
        self._elbow1 = elbow1
        self._elbow2 = elbow2
        self._leg1 = leg1
        self._leg2 = leg2
        self._knee1 = knee1
        self._knee2 = knee2

    @property
    def head(self) -> Coordinates:
        return self._head

    @property
    def shoulders(self) -> Coordinates:
        return self._shoulders

    @property
    def centroid(self) -> Coordinates:
        return self._centroid

    @property
    def hand1(self) -> Coordinates:
        return self._hand1

    @property
    def hand2(self) -> Coordinates:
        return self._hand2

    @property
    def elbow1(self) -> Coordinates:
        return self._elbow1

    @property
    def elbow2(self) -> Coordinates:
        return self._elbow2

    @property
    def leg1(self) -> Coordinates:
        return self._leg1

    @property
    def leg2(self) -> Coordinates:
        return self._leg2

    @property
    def knee1(self) -> Coordinates:
        return self._knee1

    @property
    def knee2(self) -> Coordinates:
        return self._knee2

    @head.setter
    def head(self, h: Coordinates) -> None:
        self._head = h

    @shoulders.setter
    def shoulders(self, s: Coordinates) -> None:
        self._shoulders = s

    @centroid.setter
    def centroid(self, c: Coordinates) -> None:
        self._centroid = c

    @hand1.setter
    def hand1(self, h: Coordinates) -> None:
        self._hand1 = h

    @hand2.setter
    def hand2(self, h: Coordinates) -> None:
        self._hand2 = h

    @elbow1.setter
    def elbow1(self, e: Coordinates) -> None:
        self._elbow1 = e

    @elbow2.setter
    def elbow2(self, e: Coordinates) -> None:
        self._elbow2 = e

    @leg1.setter
    def leg1(self, leg: Coordinates) -> None:
        self._leg1 = leg

    @leg2.setter
    def leg2(self, leg: Coordinates) -> None:
        self._leg2 = leg

    @knee1.setter
    def knee1(self, k: Coordinates) -> None:
        self._knee1 = k

    @knee2.setter
    def knee2(self, k: Coordinates) -> None:
        self._knee2 = k


@dataclass
class ROI:
    def __init__(self, player, ROI_index, box_width, box_height, box_upper_left_x, box_upper_left_y, collision,
                 possible_collision):
        self._player = player
        self._ROI_index = ROI_index
        self._box_width = box_width
        self._box_height = box_height
        self._box_upper_left_x = box_upper_left_x
        self._box_upper_left_y = box_upper_left_y
        self._collision = collision
        self._possible_collision = possible_collision
        self._possible_collision_participants = []

    @property
    def player(self) -> Player:
        return self._player

    @property
    def ROI_index(self) -> int:
        return self._ROI_index

    @property
    def box_width(self) -> int:
        return self._box_width

    @property
    def box_height(self) -> int:
        return self._box_height

    @property
    def box_upper_left_x(self) -> int:
        return self._box_upper_left_x

    @property
    def box_upper_left_y(self) -> int:
        return self._box_upper_left_y

    @property
    def collision(self) -> bool:
        return self._collision

    @property
    def possible_collision(self) -> bool:
        return self._possible_collision

    @property
    def possible_collision_participants(self):
        return self._possible_collision_participants

    @player.setter
    def player(self, p: Player) -> None:
        self._player = p

    @ROI_index.setter
    def ROI_index(self, i: int) -> None:
        self._ROI_index = i

    @box_width.setter
    def box_width(self, w: int) -> None:
        self._box_width = w

    @box_height.setter
    def box_height(self, h: int) -> None:
        self._box_height = h

    @box_upper_left_x.setter
    def box_upper_left_x(self, x: int) -> None:
        self._box_upper_left_x = x

    @box_upper_left_y.setter
    def box_upper_left_y(self, y: int) -> None:
        self._box_upper_left_y = y

    @collision.setter
    def collision(self, c: bool) -> None:
        self._collision = c

    @possible_collision.setter
    def possible_collision(self, c: bool) -> None:
        self._possible_collision = c

    @possible_collision_participants.setter
    def possible_collision_participants(self, p) -> None:
        if any(type(x) != 'int' for x in p):
            raise ValueError('Only integer indexes allowed')
        self._possible_collision_participants = p