Use MAX_STATIC_DATA of 500000.
When play begins, seed the random-number generator with 1234.

container is a kind of thing.
door is a kind of thing.
object-like is a kind of thing.
supporter is a kind of thing.
food is a kind of object-like.
key is a kind of object-like.
containers are openable, lockable and fixed in place. containers are usually closed.
door is openable and lockable.
object-like is portable.
supporters are fixed in place.
food is edible.
A room has a text called internal name.


The r_1 and the r_0 and the r_4 and the r_3 and the r_6 and the r_5 and the r_2 are rooms.

Understand "still cubicle" as r_1.
The internal name of r_1 is "still cubicle".
The printed name of r_1 is "-= Still Cubicle =-".
The still cubicle part 0 is some text that varies. The still cubicle part 0 is "You find yourself in a cubicle. A still kind of place. You start to take note of what's in the room.

 Look out! It's a- oh, never mind, it's just a gross shelf. [if there is something on the s_0]On the gross shelf you can see [a list of things on the s_0].[end if]".
The still cubicle part 1 is some text that varies. The still cubicle part 1 is "[if there is nothing on the s_0]But the thing is empty, unfortunately.[end if]".
The still cubicle part 2 is some text that varies. The still cubicle part 2 is " You can make out a bookshelf. The bookshelf is worn.[if there is something on the s_1] On the worn bookshelf you can make out [a list of things on the s_1]. Suddenly, you bump your head on the ceiling, but it's not such a bad bump that it's going to prevent you from looking at objects and even things.[end if]".
The still cubicle part 3 is some text that varies. The still cubicle part 3 is "[if there is nothing on the s_1] But there isn't a thing on it. Oh! Why couldn't there just be stuff on it?[end if]".
The still cubicle part 4 is some text that varies. The still cubicle part 4 is "

There is an unblocked exit to the north. There is an exit to the west. Don't worry, it is unblocked.".
The description of r_1 is "[still cubicle part 0][still cubicle part 1][still cubicle part 2][still cubicle part 3][still cubicle part 4]".

The r_0 is mapped west of r_1.
The r_5 is mapped north of r_1.
Understand "sleepy chamber" as r_0.
The internal name of r_0 is "sleepy chamber".
The printed name of r_0 is "-= Sleepy Chamber =-".
The sleepy chamber part 0 is some text that varies. The sleepy chamber part 0 is "You find yourself in a chamber. A sleepy kind of place.

 You can see [if c_1 is locked]a locked[else if c_1 is open]an opened[otherwise]a closed[end if]".
The sleepy chamber part 1 is some text that varies. The sleepy chamber part 1 is " chest.[if c_1 is open and there is something in the c_1] The blue chest contains [a list of things in the c_1]. You can't wait to tell the folks at home about this![end if]".
The sleepy chamber part 2 is some text that varies. The sleepy chamber part 2 is "[if c_1 is open and the c_1 contains nothing] The chest is empty! This is the worst thing that could possibly happen, ever![end if]".
The sleepy chamber part 3 is some text that varies. The sleepy chamber part 3 is "

There is an exit to the east. Don't worry, it is unguarded. There is an exit to the north. Don't worry, it is unblocked.".
The description of r_0 is "[sleepy chamber part 0][sleepy chamber part 1][sleepy chamber part 2][sleepy chamber part 3]".

The r_2 is mapped north of r_0.
The r_1 is mapped east of r_0.
Understand "cozy bedchamber" as r_4.
The internal name of r_4 is "cozy bedchamber".
The printed name of r_4 is "-= Cozy Bedchamber =-".
The cozy bedchamber part 0 is some text that varies. The cozy bedchamber part 0 is "Ah, the bedchamber. This is some kind of bedchamber, really great cozy vibes in this place, a wonderful cozy atmosphere. And now, well, you're in it.



There is an exit to the south. Don't worry, it is unguarded. There is an unguarded exit to the west.".
The description of r_4 is "[cozy bedchamber part 0]".

The r_3 is mapped west of r_4.
The r_5 is mapped south of r_4.
Understand "cramped spare room" as r_3.
The internal name of r_3 is "cramped spare room".
The printed name of r_3 is "-= Cramped Spare Room =-".
The cramped spare room part 0 is some text that varies. The cramped spare room part 0 is "You find yourself in a spare room. A cramped one.

 You make out a splintery counter. [if there is something on the s_2]On the splintery counter you make out [a list of things on the s_2].[end if]".
The cramped spare room part 1 is some text that varies. The cramped spare room part 1 is "[if there is nothing on the s_2]However, the counter, like an empty counter, has nothing on it.[end if]".
The cramped spare room part 2 is some text that varies. The cramped spare room part 2 is " If you haven't noticed it already, there seems to be something there by the wall, it's a rough shelf. [if there is something on the s_3]You see [a list of things on the s_3] on the shelf.[end if]".
The cramped spare room part 3 is some text that varies. The cramped spare room part 3 is "[if there is nothing on the s_3]However, the shelf, like an empty shelf, has nothing on it.[end if]".
The cramped spare room part 4 is some text that varies. The cramped spare room part 4 is "

You don't like doors? Why not try going east, that entranceway is unblocked. There is an exit to the south. Don't worry, it is unblocked.".
The description of r_3 is "[cramped spare room part 0][cramped spare room part 1][cramped spare room part 2][cramped spare room part 3][cramped spare room part 4]".

The r_2 is mapped south of r_3.
The r_4 is mapped east of r_3.
Understand "spotless washroom" as r_6.
The internal name of r_6 is "spotless washroom".
The printed name of r_6 is "-= Spotless Washroom =-".
The spotless washroom part 0 is some text that varies. The spotless washroom part 0 is "You've just shown up in a spotless washroom. You start to take note of what's in the room.

 You smell a wretched smell, and follow it to a drawer.[if c_0 is open and there is something in the c_0] The balsamwood drawer contains [a list of things in the c_0].[end if]".
The spotless washroom part 1 is some text that varies. The spotless washroom part 1 is "[if c_0 is open and the c_0 contains nothing] The drawer is empty! What a waste of a day![end if]".
The spotless washroom part 2 is some text that varies. The spotless washroom part 2 is "

You need an unguarded exit? You should try going west.".
The description of r_6 is "[spotless washroom part 0][spotless washroom part 1][spotless washroom part 2]".

The r_5 is mapped west of r_6.
Understand "spotless steam room" as r_5.
The internal name of r_5 is "spotless steam room".
The printed name of r_5 is "-= Spotless Steam Room =-".
The spotless steam room part 0 is some text that varies. The spotless steam room part 0 is "You are in a steam room. A spotless one. You start to take note of what's in the room.



There is an unguarded exit to the east. You need an unblocked exit? You should try going north. There is an exit to the south. Don't worry, it is unblocked.".
The description of r_5 is "[spotless steam room part 0]".

The r_1 is mapped south of r_5.
The r_4 is mapped north of r_5.
The r_6 is mapped east of r_5.
Understand "silent workshop" as r_2.
The internal name of r_2 is "silent workshop".
The printed name of r_2 is "-= Silent Workshop =-".
The silent workshop part 0 is some text that varies. The silent workshop part 0 is "A decrepit notice tells you that you are now in the silent workshop.



You don't like doors? Why not try going north, that entranceway is unblocked. There is an exit to the south. Don't worry, it is unguarded.".
The description of r_2 is "[silent workshop part 0]".

The r_0 is mapped south of r_2.
The r_3 is mapped north of r_2.

The c_0 and the c_1 are containers.
The c_0 and the c_1 are privately-named.
The f_0 and the f_1 are foods.
The f_0 and the f_1 are privately-named.
The k_0 are keys.
The k_0 are privately-named.
The o_2 and the o_0 and the o_1 are object-likes.
The o_2 and the o_0 and the o_1 are privately-named.
The r_1 and the r_0 and the r_4 and the r_3 and the r_6 and the r_5 and the r_2 are rooms.
The r_1 and the r_0 and the r_4 and the r_3 and the r_6 and the r_5 and the r_2 are privately-named.
The s_0 and the s_1 and the s_2 and the s_3 are supporters.
The s_0 and the s_1 and the s_2 and the s_3 are privately-named.

The description of c_0 is "The balsamwood drawer looks strong, and impossible to destroy. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_0 is "balsamwood drawer".
Understand "balsamwood drawer" as c_0.
Understand "balsamwood" as c_0.
Understand "drawer" as c_0.
The c_0 is in r_6.
The c_0 is open.
The description of c_1 is "The blue chest looks strong, and impossible to break. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_1 is "blue chest".
Understand "blue chest" as c_1.
Understand "blue" as c_1.
Understand "chest" as c_1.
The c_1 is in r_0.
The c_1 is closed.
The description of f_0 is "You couldn't pay me to eat that frozen thing.".
The printed name of f_0 is "frozen potato".
Understand "frozen potato" as f_0.
Understand "frozen" as f_0.
Understand "potato" as f_0.
The f_0 is in r_1.
The f_0 is edible.
The description of k_0 is "The blue key is light.".
The printed name of k_0 is "blue key".
Understand "blue key" as k_0.
Understand "blue" as k_0.
Understand "key" as k_0.
The k_0 is in r_0.
The matching key of the c_1 is the k_0.
The description of o_2 is "The grotesque butterfly is brand new.".
The printed name of o_2 is "grotesque butterfly".
Understand "grotesque butterfly" as o_2.
Understand "grotesque" as o_2.
Understand "butterfly" as o_2.
The o_2 is in r_4.
The description of s_0 is "The gross shelf is stable.".
The printed name of s_0 is "gross shelf".
Understand "gross shelf" as s_0.
Understand "gross" as s_0.
Understand "shelf" as s_0.
The s_0 is in r_1.
The description of s_1 is "The worn bookshelf is an unstable piece of trash.".
The printed name of s_1 is "worn bookshelf".
Understand "worn bookshelf" as s_1.
Understand "worn" as s_1.
Understand "bookshelf" as s_1.
The s_1 is in r_1.
The description of s_2 is "The splintery counter is wobbly.".
The printed name of s_2 is "splintery counter".
Understand "splintery counter" as s_2.
Understand "splintery" as s_2.
Understand "counter" as s_2.
The s_2 is in r_3.
The description of s_3 is "The rough shelf is unstable.".
The printed name of s_3 is "rough shelf".
Understand "rough shelf" as s_3.
Understand "rough" as s_3.
Understand "shelf" as s_3.
The s_3 is in r_3.
The description of f_1 is "that's a half-eaten potato!".
The printed name of f_1 is "half-eaten potato".
Understand "half-eaten potato" as f_1.
Understand "half-eaten" as f_1.
Understand "potato" as f_1.
The f_1 is edible.
The player carries the f_1.
The description of o_0 is "The torn paper towel is modern.".
The printed name of o_0 is "torn paper towel".
Understand "torn paper towel" as o_0.
Understand "torn" as o_0.
Understand "paper" as o_0.
Understand "towel" as o_0.
The o_0 is in the c_0.
The description of o_1 is "The operational mouse is clean.".
The printed name of o_1 is "operational mouse".
Understand "operational mouse" as o_1.
Understand "operational" as o_1.
Understand "mouse" as o_1.
The o_1 is on the s_1.


The player is in r_1.

The quest0 completed is a truth state that varies.
The quest0 completed is usually false.

Test quest0_0 with "go west / go north / go north / go east / go south / go east / take torn paper towel from balsamwood drawer / go west / go south / drop torn paper towel"

Every turn:
	if quest0 completed is true:
		do nothing;
	else if The player is in r_1 and The o_0 is in r_1:
		increase the score by 1; [Quest completed]
		if 1 is 1 [always true]:
			Now the quest0 completed is true;

Use scoring. The maximum score is 1.
This is the simpler notify score changes rule:
	If the score is not the last notified score:
		let V be the score - the last notified score;
		if V > 0:
			say "Your score has just gone up by [V in words] ";
		else:
			say "Your score changed by [V in words] ";
		if V >= -1 and V <= 1:
			say "point.";
		else:
			say "points.";
		Now the last notified score is the score;
	if quest0 completed is true:
		end the story finally; [Win]

The simpler notify score changes rule substitutes for the notify score changes rule.

Rule for listing nondescript items:
	stop.

Rule for printing the banner text:
	say "[fixed letter spacing]";
	say "                    ________  ________  __    __  ________        [line break]";
	say "                   |        \|        \|  \  |  \|        \       [line break]";
	say "                    \$$$$$$$$| $$$$$$$$| $$  | $$ \$$$$$$$$       [line break]";
	say "                      | $$   | $$__     \$$\/  $$   | $$          [line break]";
	say "                      | $$   | $$  \     >$$  $$    | $$          [line break]";
	say "                      | $$   | $$$$$    /  $$$$\    | $$          [line break]";
	say "                      | $$   | $$_____ |  $$ \$$\   | $$          [line break]";
	say "                      | $$   | $$     \| $$  | $$   | $$          [line break]";
	say "                       \$$    \$$$$$$$$ \$$   \$$    \$$          [line break]";
	say "              __       __   ______   _______   __        _______  [line break]";
	say "             |  \  _  |  \ /      \ |       \ |  \      |       \ [line break]";
	say "             | $$ / \ | $$|  $$$$$$\| $$$$$$$\| $$      | $$$$$$$\[line break]";
	say "             | $$/  $\| $$| $$  | $$| $$__| $$| $$      | $$  | $$[line break]";
	say "             | $$  $$$\ $$| $$  | $$| $$    $$| $$      | $$  | $$[line break]";
	say "             | $$ $$\$$\$$| $$  | $$| $$$$$$$\| $$      | $$  | $$[line break]";
	say "             | $$$$  \$$$$| $$__/ $$| $$  | $$| $$_____ | $$__/ $$[line break]";
	say "             | $$$    \$$$ \$$    $$| $$  | $$| $$     \| $$    $$[line break]";
	say "              \$$      \$$  \$$$$$$  \$$   \$$ \$$$$$$$$ \$$$$$$$ [line break]";
	say "[variable letter spacing][line break]";
	say "[objective][line break]".

Include Basic Screen Effects by Emily Short.

Rule for printing the player's obituary:
	if story has ended finally:
		center "*** The End ***";
	else:
		center "*** You lost! ***";
	say paragraph break;
	if maximum score is -32768:
		say "You scored a total of [score] point[s], in [turn count] turn[s].";
	else:
		say "You scored [score] out of a possible [maximum score], in [turn count] turn[s].";
	[wait for any key;
	stop game abruptly;]
	rule succeeds.

Carry out requesting the score:
	if maximum score is -32768:
		say "You have so far scored [score] point[s], in [turn count] turn[s].";
	else:
		say "You have so far scored [score] out of a possible [maximum score], in [turn count] turn[s].";
	rule succeeds.

Rule for implicitly taking something (called target):
	if target is fixed in place:
		say "The [target] is fixed in place.";
	otherwise:
		say "You need to take the [target] first.";
		set pronouns from target;
	stop.

Does the player mean doing something:
	if the noun is not nothing and the second noun is nothing and the player's command matches the text printed name of the noun:
		it is likely;
	if the noun is nothing and the second noun is not nothing and the player's command matches the text printed name of the second noun:
		it is likely;
	if the noun is not nothing and the second noun is not nothing and the player's command matches the text printed name of the noun and the player's command matches the text printed name of the second noun:
		it is very likely.  [Handle action with two arguments.]

Printing the content of the room is an activity.
Rule for printing the content of the room:
	let R be the location of the player;
	say "Room contents:[line break]";
	list the contents of R, with newlines, indented, including all contents, with extra indentation.

Printing the content of the world is an activity.
Rule for printing the content of the world:
	let L be the list of the rooms;
	say "World: [line break]";
	repeat with R running through L:
		say "  [the internal name of R][line break]";
	repeat with R running through L:
		say "[the internal name of R]:[line break]";
		if the list of things in R is empty:
			say "  nothing[line break]";
		otherwise:
			list the contents of R, with newlines, indented, including all contents, with extra indentation.

Printing the content of the inventory is an activity.
Rule for printing the content of the inventory:
	say "You are carrying: ";
	list the contents of the player, as a sentence, giving inventory information, including all contents;
	say ".".

The print standard inventory rule is not listed in any rulebook.
Carry out taking inventory (this is the new print inventory rule):
	say "You are carrying: ";
	list the contents of the player, as a sentence, giving inventory information, including all contents;
	say ".".

Printing the content of nowhere is an activity.
Rule for printing the content of nowhere:
	say "Nowhere:[line break]";
	let L be the list of the off-stage things;
	repeat with thing running through L:
		say "  [thing][line break]";

Printing the things on the floor is an activity.
Rule for printing the things on the floor:
	let R be the location of the player;
	let L be the list of things in R;
	remove yourself from L;
	remove the list of containers from L;
	remove the list of supporters from L;
	remove the list of doors from L;
	if the number of entries in L is greater than 0:
		say "There is [L with indefinite articles] on the floor.";

After printing the name of something (called target) while
printing the content of the room
or printing the content of the world
or printing the content of the inventory
or printing the content of nowhere:
	follow the property-aggregation rules for the target.

The property-aggregation rules are an object-based rulebook.
The property-aggregation rulebook has a list of text called the tagline.

[At the moment, we only support "open/unlocked", "closed/unlocked" and "closed/locked" for doors and containers.]
[A first property-aggregation rule for an openable open thing (this is the mention open openables rule):
	add "open" to the tagline.

A property-aggregation rule for an openable closed thing (this is the mention closed openables rule):
	add "closed" to the tagline.

A property-aggregation rule for an lockable unlocked thing (this is the mention unlocked lockable rule):
	add "unlocked" to the tagline.

A property-aggregation rule for an lockable locked thing (this is the mention locked lockable rule):
	add "locked" to the tagline.]

A first property-aggregation rule for an openable lockable open unlocked thing (this is the mention open openables rule):
	add "open" to the tagline.

A property-aggregation rule for an openable lockable closed unlocked thing (this is the mention closed openables rule):
	add "closed" to the tagline.

A property-aggregation rule for an openable lockable closed locked thing (this is the mention locked openables rule):
	add "locked" to the tagline.

A property-aggregation rule for a lockable thing (called the lockable thing) (this is the mention matching key of lockable rule):
	let X be the matching key of the lockable thing;
	if X is not nothing:
		add "match [X]" to the tagline.

A property-aggregation rule for an edible off-stage thing (this is the mention eaten edible rule):
	add "eaten" to the tagline.

The last property-aggregation rule (this is the print aggregated properties rule):
	if the number of entries in the tagline is greater than 0:
		say " ([tagline])";
		rule succeeds;
	rule fails;

The objective part 0 is some text that varies. The objective part 0 is "You are now playing a life changing game of TextWorld! Here is your task for today. First step, move west. Following that, move north. If you can finish that, venture north. With that accomplished, at".
The objective part 1 is some text that varies. The objective part 1 is "tempt to venture east. Then, head south. And then, travel east. And then, pick up the torn paper towel from the balsamwood drawer. Then, try to take a trip west. Then, try to move south. Then, ditch t".
The objective part 2 is some text that varies. The objective part 2 is "he torn paper towel on the floor of the still cubicle. Got that? Good!".

An objective is some text that varies. The objective is "[objective part 0][objective part 1][objective part 2]".
Printing the objective is an action applying to nothing.
Carry out printing the objective:
	say "[objective]".

Understand "goal" as printing the objective.

The taking action has an object called previous locale (matched as "from").

Setting action variables for taking:
	now previous locale is the holder of the noun.

Report taking something from the location:
	say "You pick up [the noun] from the ground." instead.

Report taking something:
	say "You take [the noun] from [the previous locale]." instead.

Report dropping something:
	say "You drop [the noun] on the ground." instead.

The print state option is a truth state that varies.
The print state option is usually false.

Turning on the print state option is an action applying to nothing.
Carry out turning on the print state option:
	Now the print state option is true.

Turning off the print state option is an action applying to nothing.
Carry out turning off the print state option:
	Now the print state option is false.

Printing the state is an activity.
Rule for printing the state:
	let R be the location of the player;
	say "Room: [line break] [the internal name of R][line break]";
	[say "[line break]";
	carry out the printing the content of the room activity;]
	say "[line break]";
	carry out the printing the content of the world activity;
	say "[line break]";
	carry out the printing the content of the inventory activity;
	say "[line break]";
	carry out the printing the content of nowhere activity;
	say "[line break]".

Printing the entire state is an action applying to nothing.
Carry out printing the entire state:
	say "-=STATE START=-[line break]";
	carry out the printing the state activity;
	say "[line break]Score:[line break] [score]/[maximum score][line break]";
	say "[line break]Objective:[line break] [objective][line break]";
	say "[line break]Inventory description:[line break]";
	say "  You are carrying: [a list of things carried by the player].[line break]";
	say "[line break]Room description:[line break]";
	try looking;
	say "[line break]-=STATE STOP=-";

Every turn:
	if extra description command option is true:
		say "<description>";
		try looking;
		say "</description>";
	if extra inventory command option is true:
		say "<inventory>";
		try taking inventory;
		say "</inventory>";
	if extra score command option is true:
		say "<score>[line break][score][line break]</score>";
	if extra score command option is true:
		say "<moves>[line break][turn count][line break]</moves>";
	if print state option is true:
		try printing the entire state;

When play ends:
	if print state option is true:
		try printing the entire state;

After looking:
	carry out the printing the things on the floor activity.

Understand "print_state" as printing the entire state.
Understand "enable print state option" as turning on the print state option.
Understand "disable print state option" as turning off the print state option.

Before going through a closed door (called the blocking door):
	say "You have to open the [blocking door] first.";
	stop.

Before opening a locked door (called the locked door):
	let X be the matching key of the locked door;
	if X is nothing:
		say "The [locked door] is welded shut.";
	otherwise:
		say "You have to unlock the [locked door] with the [X] first.";
	stop.

Before opening a locked container (called the locked container):
	let X be the matching key of the locked container;
	if X is nothing:
		say "The [locked container] is welded shut.";
	otherwise:
		say "You have to unlock the [locked container] with the [X] first.";
	stop.

Displaying help message is an action applying to nothing.
Carry out displaying help message:
	say "[fixed letter spacing]Available commands:[line break]";
	say "  look:                describe the current room[line break]";
	say "  goal:                print the goal of this game[line break]";
	say "  inventory:           print player's inventory[line break]";
	say "  go <dir>:            move the player north, east, south or west[line break]";
	say "  examine ...:         examine something more closely[line break]";
	say "  eat ...:             eat edible food[line break]";
	say "  open ...:            open a door or a container[line break]";
	say "  close ...:           close a door or a container[line break]";
	say "  drop ...:            drop an object on the floor[line break]";
	say "  take ...:            take an object that is on the floor[line break]";
	say "  put ... on ...:      place an object on a supporter[line break]";
	say "  take ... from ...:   take an object from a container or a supporter[line break]";
	say "  insert ... into ...: place an object into a container[line break]";
	say "  lock ... with ...:   lock a door or a container with a key[line break]";
	say "  unlock ... with ...: unlock a door or a container with a key[line break]";

Understand "help" as displaying help message.

Taking all is an action applying to nothing.
Check taking all:
	say "You have to be more specific!";
	rule fails.

Understand "take all" as taking all.
Understand "get all" as taking all.
Understand "pick up all" as taking all.

Understand "take each" as taking all.
Understand "get each" as taking all.
Understand "pick up each" as taking all.

Understand "take everything" as taking all.
Understand "get everything" as taking all.
Understand "pick up everything" as taking all.

The extra description command option is a truth state that varies.
The extra description command option is usually false.

Turning on the extra description command option is an action applying to nothing.
Carry out turning on the extra description command option:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	Now the extra description command option is true.

Understand "tw-extra-infos description" as turning on the extra description command option.

The extra inventory command option is a truth state that varies.
The extra inventory command option is usually false.

Turning on the extra inventory command option is an action applying to nothing.
Carry out turning on the extra inventory command option:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	Now the extra inventory command option is true.

Understand "tw-extra-infos inventory" as turning on the extra inventory command option.

The extra score command option is a truth state that varies.
The extra score command option is usually false.

Turning on the extra score command option is an action applying to nothing.
Carry out turning on the extra score command option:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	Now the extra score command option is true.

Understand "tw-extra-infos score" as turning on the extra score command option.

The extra moves command option is a truth state that varies.
The extra moves command option is usually false.

Turning on the extra moves command option is an action applying to nothing.
Carry out turning on the extra moves command option:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	Now the extra moves command option is true.

Understand "tw-extra-infos moves" as turning on the extra moves command option.

To trace the actions:
	(- trace_actions = 1; -).

Tracing the actions is an action applying to nothing.
Carry out tracing the actions:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	trace the actions;

Understand "tw-trace-actions" as tracing the actions.

The restrict commands option is a truth state that varies.
The restrict commands option is usually false.

Turning on the restrict commands option is an action applying to nothing.
Carry out turning on the restrict commands option:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	Now the restrict commands option is true.

Understand "restrict commands" as turning on the restrict commands option.

The taking allowed flag is a truth state that varies.
The taking allowed flag is usually false.

Before removing something from something:
	now the taking allowed flag is true.

After removing something from something:
	now the taking allowed flag is false.

Before taking a thing (called the object) when the object is on a supporter (called the supporter):
	if the restrict commands option is true and taking allowed flag is false:
		say "Can't see any [object] on the floor! Try taking the [object] from the [supporter] instead.";
		rule fails.

Before of taking a thing (called the object) when the object is in a container (called the container):
	if the restrict commands option is true and taking allowed flag is false:
		say "Can't see any [object] on the floor! Try taking the [object] from the [container] instead.";
		rule fails.

Understand "take [something]" as removing it from.

Rule for supplying a missing second noun while removing:
	if restrict commands option is false and noun is on a supporter (called the supporter):
		now the second noun is the supporter;
	else if restrict commands option is false and noun is in a container (called the container):
		now the second noun is the container;
	else:
		try taking the noun;
		say ""; [Needed to avoid printing a default message.]

The version number is always 1.

Reporting the version number is an action applying to nothing.
Carry out reporting the version number:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	say "[version number]".

Understand "tw-print version" as reporting the version number.

Reporting max score is an action applying to nothing.
Carry out reporting max score:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	if maximum score is -32768:
		say "infinity";
	else:
		say "[maximum score]".

Understand "tw-print max_score" as reporting max score.

To print id of (something - thing):
	(- print {something}, "^"; -).

Printing the id of player is an action applying to nothing.
Carry out printing the id of player:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	print id of player.

Printing the id of EndOfObject is an action applying to nothing.
Carry out printing the id of EndOfObject:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	print id of EndOfObject.

Understand "tw-print player id" as printing the id of player.
Understand "tw-print EndOfObject id" as printing the id of EndOfObject.

There is a EndOfObject.

